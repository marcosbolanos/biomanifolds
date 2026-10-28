"""Steering of the helix face angle (notebook 5): h <- h - f(own angle) + f(target angle) at the masked residue, at the steering
layer, f the harmonic fit; 4 source bins, 72 target angles, and random-vector controls of the same norm.

Results in out/steering/results/source_*.parquet (one per source bin; finished files are skipped), figures with prefix "steering_".
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from tqdm.auto import tqdm

from biomanifolds import theme
from biomanifolds.angle_study.constants import AA, CONTROL_OFFSETS, EISENBERG, HYDROPHOBIC, SOURCE_ANGLES, SOURCE_BIN_WIDTH, TARGETS
from biomanifolds.angle_study.figures import save, wheel

EDITS = ["none"] + [f"target {t}" for t in TARGETS] + [f"random vector {o}" for o in CONTROL_OFFSETS]
EDIT_INFO = pl.DataFrame({
    "edit": EDITS,
    "kind": ["none"] + ["steer"] * len(TARGETS) + ["random vector"] * len(CONTROL_OFFSETS),
    "target_angle": [None] + [float(t) for t in TARGETS] + [None] * len(CONTROL_OFFSETS),
    "control_offset": [None] * (1 + len(TARGETS)) + CONTROL_OFFSETS,
}, schema_overrides={"target_angle": pl.Float64, "control_offset": pl.Int64})


def harmonic(fits, t):
    """f(t) for angles t (radians, any shape): (..., hidden)."""
    t = np.asarray(t)
    return np.stack([np.ones_like(t), np.cos(t), np.sin(t)], -1) @ fits["harmonic"]


def edit_vectors(fits, n, own, hidden_size):
    """(edits, hidden): zero, f(target) - f(own) for every target, and the random-vector controls.

    One random generator per residue (n: its row in the sample); draws: for each control offset, one random direction."""
    residue_rng = np.random.default_rng([0, n])
    own = np.deg2rad(own)
    steer = harmonic(fits, np.deg2rad(TARGETS)) - harmonic(fits, own)
    controls = []
    for offset in CONTROL_OFFSETS:
        direction = residue_rng.standard_normal(hidden_size)
        norm = np.linalg.norm(harmonic(fits, own + np.deg2rad(offset)) - harmonic(fits, own))
        controls.append(direction / np.linalg.norm(direction) * norm)
    return np.concatenate([np.zeros((1, hidden_size)), steer, np.stack(controls)])


def run(config, model, residues, counts, out):
    """Every edit for every source residue (accession, i, helix_face, source_angle, sequence)."""
    directory, figures = out / "steering", out / "figures"
    (directory / "results").mkdir(parents=True, exist_ok=True)
    counts.write_parquet(directory / "sample_counts.parquet")
    print(counts, flush=True)
    figure_sample(residues, figures)

    layer = config["steering_layer"]
    fits = dict(np.load(out / "bins" / f"fits_L{layer}.npz"))
    aa_ids = model.ids(AA)
    numbered = residues.with_row_index("n")
    for source in SOURCE_ANGLES:
        path = directory / "results" / f"source_{source}.parquet"
        if path.exists():
            continue
        blocks = []
        source_residues = numbered.filter(pl.col("source_angle") == source)
        for n, i, own, sequence in tqdm(source_residues.select("n", "i", "helix_face", "sequence").iter_rows(), total=len(source_residues),
                                        desc=f"steering, source {source}", mininterval=30):
            input_ids, _ = model.token_ids([sequence])
            position = i + model.offset
            input_ids[0, position] = model.mask_id
            deltas = edit_vectors(fits, n, own, model.hidden_size)
            h = model.run(model.embed(input_ids), 0, layer)  # unedited, up to the steering layer
            edited = h.repeat(len(EDITS), 1, 1)  # one copy per edit
            edited[:, position] += torch.tensor(deltas, dtype=edited.dtype, device=model.device)
            edited = model.run(edited, layer, model.n_layers)
            logits = model.head(edited[:, position])  # (edits, vocabulary)
            blocks.append(pl.DataFrame({
                "n": n, "layer": layer, "edit": EDITS, "edit_norm": np.linalg.norm(deltas, axis=1),
                "logits": logits.float().cpu().numpy(),  # all vocabulary logits, in the order of the vocabulary
                "aa_log_probs": logits.log_softmax(-1)[:, aa_ids].float().cpu().numpy(),  # the 20 amino acids, in the order of AA
            }))
        rows = (
            pl.concat(blocks)
            .join(numbered, on="n", maintain_order="left")
            .join(EDIT_INFO, on="edit", maintain_order="left")
            .with_columns(aa=pl.col("sequence").str.slice(pl.col("i"), 1))
            .select("accession", "i", "aa", "helix_face", "source_angle", "layer", "edit", "kind", "target_angle", "control_offset",
                    "edit_norm", "logits", "aa_log_probs")
        )
        rows.write_parquet(path.with_suffix(".tmp"))
        path.with_suffix(".tmp").rename(path)  # a results file exists only when it is complete
    (directory / "metadata.json").write_text(json.dumps({
        "model": config["hf_id"], "layer": layer, "source_angles": SOURCE_ANGLES, "bin_width": SOURCE_BIN_WIDTH,
        "targets": TARGETS, "control_offsets": CONTROL_OFFSETS,
        "edit": "h <- h - f(own) + f(target) at the masked residue, f: harmonic fit on the train-set bin means",
        "logits_order": model.vocabulary, "aa_log_probs_order": AA,
    }, indent=1))


def figure_sample(residues, figures):
    fig, axes = plt.subplots(1, len(SOURCE_ANGLES), figsize=(16, 3), sharey=True)
    for ax, source in zip(axes, SOURCE_ANGLES):
        own = residues.filter(pl.col("source_angle") == source)["helix_face"].to_numpy()
        ax.hist((own - source + 180) % 360 - 180, bins=np.linspace(-SOURCE_BIN_WIDTH / 2, SOURCE_BIN_WIDTH / 2, 26))
        ax.set_title(f"source {source}°")
        ax.set_xlabel("own angle - source angle (degrees)")
    axes[0].set_ylabel("residues")
    fig.tight_layout()
    save(fig, figures, "steering_source_angles")


def analyse(config, out, n_per_bin):
    """Changes of the hydrophobic probability mass and of E[H] per target angle; wheels, curves and canonical examples."""
    directory, figures = out / "steering", out / "figures"
    results = pl.read_parquet(directory / "results" / "*.parquet")
    eisenberg = np.array([EISENBERG[a] for a in AA])

    # probabilities over the 20 amino acids (renormalised), then the two scores per row
    log_probs = results["aa_log_probs"].to_numpy()
    p = np.exp(log_probs - log_probs.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    scores = results.select("accession", "i", "source_angle", "layer", "edit", "kind", "target_angle", "control_offset").with_columns(
        hydrophobic_mass=p[:, [a in HYDROPHOBIC for a in AA]].sum(1), expected_hydrophobicity=p @ eisenberg)
    unedited = scores.filter(pl.col("edit") == "none").select("accession", "i", "layer", mass_unedited="hydrophobic_mass", h_unedited="expected_hydrophobicity")
    changes = scores.filter(pl.col("edit") != "none").join(unedited, on=["accession", "i", "layer"]).with_columns(
        delta_mass=pl.col("hydrophobic_mass") - pl.col("mass_unedited"), delta_h=pl.col("expected_hydrophobicity") - pl.col("h_unedited"))
    # one residue per protein in each source bin: the residues are independent, 95% CI from the standard error
    per_edit = changes.group_by("source_angle", "layer", "edit", "kind", "target_angle", "control_offset").agg(
        n=pl.len(), *[pl.col(c).mean().alias(f"mean_{c}") for c in ["delta_mass", "delta_h"]],
        *[(1.96 * pl.col(c).std() / pl.len().sqrt()).alias(f"ci_{c}") for c in ["delta_mass", "delta_h"]],
    ).sort("source_angle", "kind", "target_angle", "control_offset")
    per_edit.write_parquet(directory / "per_edit.parquet")
    comparison = (
        per_edit.filter(pl.col("kind") == "random vector").select("source_angle", "layer", "control_offset", "n",
                                                                  control_delta_mass="mean_delta_mass", control_ci="ci_delta_mass")
        .with_columns(target_angle=((pl.col("source_angle") + pl.col("control_offset") + 180) % 360 - 180).cast(pl.Float64))
        .join(per_edit.filter(pl.col("kind") == "steer").select("source_angle", "layer", "target_angle", steer_delta_mass="mean_delta_mass",
                                                                steer_ci="ci_delta_mass"), on=["source_angle", "layer", "target_angle"])
        .sort("layer", "source_angle", "control_offset")
    )
    comparison.write_parquet(directory / "steering_vs_random_vector.parquet")
    print(comparison, flush=True)
    layer = config["steering_layer"]

    for column, label, title in [
        ("delta_mass", "change of the hydrophobic probability mass", "change of the probability mass on hydrophobic amino acids"),
        ("delta_h", "change of the expected Eisenberg hydrophobicity", "change of the expected Eisenberg hydrophobicity of the prediction"),
    ]:
        steer = per_edit.filter(pl.col("kind") == "steer")
        limit = steer[f"mean_{column}"].abs().max()
        norm = Normalize(-limit, limit)
        fig, axes = plt.subplots(1, len(SOURCE_ANGLES), figsize=(4.6 * len(SOURCE_ANGLES) + 1, 6), subplot_kw={"projection": "polar"},
                                 layout="constrained")
        fig.get_layout_engine().set(wspace=0.02, h_pad=0.15)  # wheels close together; some room below the title
        for ax, source in zip(axes, SOURCE_ANGLES):
            d = steer.filter(pl.col("source_angle") == source).sort("target_angle")
            wheel(ax, d["target_angle"].to_numpy(), d[f"mean_{column}"].to_numpy(), 5, norm, theme.DELTA, f"source {source}° (dot), layer {layer}")
            ax.plot(np.deg2rad(source), 0.8, marker="o", markersize=8, color="black")  # inside the ring, at the source angle
        fig.colorbar(ScalarMappable(norm, theme.DELTA), ax=axes, shrink=0.6, label=label)
        fig.suptitle(f"Steering to each target angle: {title}\n({n_per_bin} residues per source bin), test set", wrap=False)
        save(fig, figures, f"steering_wheels_{column}")

    # change of E[H] against the target angle, one line per source bin with its 95% CI band; a dot at the source angle
    fig, ax = plt.subplots(figsize=(12, 5))
    for k, source in enumerate(SOURCE_ANGLES):
        steer = per_edit.filter((pl.col("source_angle") == source) & (pl.col("kind") == "steer")).sort("target_angle")
        x, y, c = steer["target_angle"].to_numpy(), steer["mean_delta_h"].to_numpy(), steer["ci_delta_h"].to_numpy()
        ax.plot(x, y, color=f"C{k}", label=f"source {source}°")
        ax.fill_between(x, y - c, y + c, color=f"C{k}", alpha=0.3)
        at_source = source if source != 180 else -180  # the targets run from -180 to 175 degrees
        ax.plot(at_source, y[x == at_source][0], marker="o", markersize=8, color=f"C{k}", zorder=3)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(range(-180, 181, 45))
    ax.set_xlim(-180, 180)
    ax.set_xlabel("target angle (degrees; 0 = toward the protein centroid)")
    ax.set_ylabel("mean ΔE[H] (Eisenberg scale)")
    ax.set_title(f"layer {layer}")
    ax.legend(fontsize="small")
    fig.suptitle(f"Steering to each target angle: change of the expected Eisenberg hydrophobicity E[H] of the prediction, 95% CI "
                 f"({n_per_bin} residues per source bin), test set")
    fig.tight_layout()
    save(fig, figures, "steering_delta_h_by_target")

    canonical_examples(results, changes, layer, eisenberg, figures)


def canonical_examples(results, changes, layer, eisenberg, figures):
    """One residue of the 0° bin and one of the 180° bin, turned by 180 degrees: the residue closest to the means of its bin."""
    turn = {0: -180, 180: 0}  # source angle -> target angle of the 180-degree turn
    colour_norm = Normalize(-np.abs(eisenberg).max(), np.abs(eisenberg).max())
    examples = []
    for source, target in turn.items():
        turned = changes.filter((pl.col("source_angle") == source) & (pl.col("target_angle") == target))
        distance = sum(((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()) ** 2 for c in ["mass_unedited", "h_unedited", "delta_mass", "delta_h"])
        examples.append(turned.with_columns(distance=distance).sort("distance").row(0, named=True) | {"target": target})
    fig, axes = plt.subplots(len(examples), 2, figsize=(16, 7.5 * len(examples)), sharex="row")
    fig.subplots_adjust(wspace=0.3, hspace=0.35)
    for row, example in zip(axes, examples):
        residue = results.filter((pl.col("accession") == example["accession"]) & (pl.col("i") == example["i"]) & (pl.col("layer") == layer))
        for ax, (name, edit) in zip(row, [("unedited", "none"), (f"steered to {example['target']}°", f"target {example['target']}")]):
            r = residue.filter(pl.col("edit") == edit).row(0, named=True)
            probs = np.exp(np.array(r["aa_log_probs"]))  # softmax over the full vocabulary
            order = np.argsort(probs)[::-1]  # highest first
            other = 1 - probs.sum()  # the other tokens of the vocabulary, together
            ax.barh(np.arange(len(AA)), probs[order], color=theme.HYDROPHOBICITY(colour_norm(eisenberg[order])))
            ax.barh(len(AA), other, color="grey")
            ax.text(other, len(AA), f" {other:.1e}", va="center", fontsize=8, fontweight=theme.WEIGHTS["note"])
            ax.set_yticks(np.arange(len(AA) + 1), [AA[k] for k in order] + ["other"])
            for tick in ax.get_yticklabels():
                tick.set_fontweight(theme.WEIGHTS["highlight"] if tick.get_text() == r["aa"] else theme.WEIGHTS["text"])
            ax.invert_yaxis()
            p_aa = probs / probs.sum()
            ax.text(0.97, 0.03, f"E[H] = {p_aa @ eisenberg:+.2f}", transform=ax.transAxes, ha="right", va="bottom",
                    bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "grey"})
            ax.set_xlabel("probability")
            ax.set_title(f"{r['accession']} residue {r['i']} (true: {r['aa']})\nown angle {r['helix_face']:.1f}°, {name}, layer {r['layer']}", wrap=False)
    fig.colorbar(ScalarMappable(colour_norm, theme.HYDROPHOBICITY), ax=axes, shrink=0.5, label="Eisenberg consensus hydrophobicity")
    fig.suptitle("Canonical examples of the 180-degree turn, test set")
    save(fig, figures, "steering_canonical_examples")
