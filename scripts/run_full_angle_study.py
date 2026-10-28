"""Run the whole helix face angle study (notebooks 1-5) for one model.

    uv run python scripts/run_full_angle_study.py configs/models/esm2_t30_150M.toml
    uv run python scripts/run_full_angle_study.py configs/models/esm2_t6_8M.toml --smoke  # small samples, a few minutes

Outputs in <angle_study path of project-config.toml>/<model name>[_smoke]/: bins/, ablation/, analysis/, steering/, figures/.
Every stage keeps its results; a stage that is already done is skipped, so the script can run again after a crash.
"""

import argparse
import functools
import json
import time

import matplotlib

matplotlib.use("Agg")  # figures are only saved

from biomanifolds import config as project_config, theme  # noqa: E402
from biomanifolds.angle_study import ablation, analysis, bins, data, samples, steering  # noqa: E402
from biomanifolds.angle_study.constants import Sizes  # noqa: E402
from biomanifolds.angle_study.models import load_config, load_model  # noqa: E402

STAGES = ["data", "bins", "ablation", "analysis", "steering"]
# settings that the saved results depend on: they must not change for an output directory. The layers can change (the bins
# and the ablation add the missing layers); batch sizes, n_boot and min_count do not change saved results.
FIXED = {"model": ["hf_id", "adapter", "steering_layer", "matmul_precision"],
         "sizes": ["train_fraction", "test_residues", "test_seed", "steering_per_bin", "steering_seed"]}


def check_saved_config(path, saved):
    """Stop if config.json of an earlier run has other values for the FIXED settings."""
    if not path.exists():
        return
    previous = json.loads(path.read_text())
    changed = [f"{part}.{key}: {previous[part].get(key)} -> {saved[part].get(key)}"
               for part, keys in FIXED.items() for key in keys if previous[part].get(key) != saved[part].get(key)]
    if changed:
        raise SystemExit(f"{path} has other settings ({'; '.join(changed)}): its saved results do not apply. "
                         "Give the model config another name, or remove the output directory.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="model config (configs/models/*.toml)")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    parser.add_argument("--smoke", action="store_true", help="small samples, to test the pipeline")
    args = parser.parse_args()

    config = load_config(args.config)
    sizes = Sizes().smoke() if args.smoke else Sizes()
    out = project_config.data_path("angle_study") / (config["name"] + ("_smoke" if args.smoke else ""))
    out.mkdir(parents=True, exist_ok=True)
    saved = {"model": config, "sizes": sizes.__dict__}
    check_saved_config(out / "config.json", saved)
    (out / "config.json").write_text(json.dumps(saved, indent=1))
    theme.use()
    print(f"{config['name']}: stages {args.stages}, outputs in {out}", flush=True)

    # the data stage prepares the splits and labels that the samples read: it runs first, and the samples load only when needed
    stages = sorted(args.stages, key=STAGES.index)

    @functools.cache
    def model():
        loaded = load_model(config)
        print(f"model {config['hf_id']}: {loaded.n_layers} layers, hidden size {loaded.hidden_size}, device {loaded.device}", flush=True)
        return loaded

    @functools.cache
    def test_split():
        test_proteins = samples.split_proteins("test")
        return test_proteins, samples.test_helix_residues(test_proteins)

    @functools.cache
    def test_residues():
        residues = samples.test_residues(*test_split(), sizes.test_residues, sizes.test_seed)
        print(f"test residues: {len(residues):,} in {residues['accession'].n_unique():,} proteins", flush=True)
        return residues

    for stage in stages:
        start = time.time()
        print(f"=== {stage}", flush=True)
        if stage == "data":
            data.run(out)
        elif stage == "bins":
            bins.run(config, model, sizes, samples.train_proteins(sizes.train_fraction), samples.residue_proteins(test_residues()), out)
        elif stage == "ablation":
            ablation.run(config, model(), test_residues(), out)
            print(ablation.summary(out, sizes.n_boot), flush=True)
        elif stage == "analysis":
            analysis.run(config, out, sizes)
        elif stage == "steering":
            residues, counts = samples.steering_residues(*test_split(), sizes.steering_per_bin, sizes.steering_seed)
            steering.run(config, model(), residues, counts, out)
            steering.analyse(config, out, sizes.steering_per_bin)
        print(f"=== {stage} done in {(time.time() - start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
