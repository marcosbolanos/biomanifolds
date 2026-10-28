"""Ablation of the helix face angle at the masked residue (notebook 3): h <- h - f(own angle) + mean, with f the bin means,
the linear fit or the harmonic fit; controls: a random angle instead of the own angle, and a random vector of the same norm.

The unedited stream runs once per batch of masked copies of a protein, to the last layer (the "none" edit, the same for every
layer); at each ablation layer the other edits branch off and run to the last layer. Results in
out/ablation/results/shard-{k:05d}-L{layer}.parquet: one file per 1000 proteins and per layer, written only when complete.
A run computes only the (shard, layer) files that are missing, so a restart or a new layer never writes a row twice.
Layers in metadata.json are done. Older runs can have files for more than one layer (shard-00000.parquet,
shard-00000-L5-6.parquet); they are read as usual, but only for layers that are done.

The random controls do not depend on the other layers or on the order of the runs: the random angle comes from
np.random.default_rng([0, n]) and the random direction of a layer and method from np.random.default_rng([0, n, layer, k]),
with n the row of the residue in the sample and k the index of the method in METHODS.
"""

import json
import re

import numpy as np
import polars as pl
import torch
from tqdm.auto import tqdm

from biomanifolds.angle_study.constants import AA, CONTROLS, METHODS

SHARD_PROTEINS = 1000
EDIT_NAMES = ["none"] + [method + control for method in METHODS for control in CONTROLS]


def load_fits(out, layers):
    """{layer: {"bin_means", "global_mean", "linear", "harmonic"}} from the bins stage."""
    return {layer: dict(np.load(out / "bins" / f"fits_L{layer}.npz")) for layer in layers}


def activation(fits, method, t):
    """Fitted mean residual stream at angle t (radians)."""
    if method == "bin_means":
        return fits["bin_means"][int(np.floor(np.rad2deg(t) + 180)) % 360]
    if method == "linear":
        return np.array([1.0, t]) @ fits["linear"]
    return np.array([1.0, np.cos(t), np.sin(t)]) @ fits["harmonic"]


def edit_vectors(fits, layers, n, face, hidden_size):
    """The changes added at the masked position, {layer: (edits, hidden)}: mean - f(theta), and the random angle (degrees).

    Seeds (n: row of the residue in the sample): the random angle from [0, n], the same for every layer and method; the
    random direction of a layer and method from [0, n, layer, k]. So an edit does not depend on the other layers of the run."""
    own, random_angle = np.deg2rad(face), np.random.default_rng([0, n]).uniform(-np.pi, np.pi)
    deltas = {}
    for layer in layers:
        d = {"none": np.zeros(hidden_size)}
        for k, method in enumerate(METHODS):
            d[method] = fits[layer]["global_mean"] - activation(fits[layer], method, own)
            d[f"{method} | random angle"] = fits[layer]["global_mean"] - activation(fits[layer], method, random_angle)
            random_direction = np.random.default_rng([0, n, layer, k]).standard_normal(hidden_size)
            d[f"{method} | random vector"] = random_direction / np.linalg.norm(random_direction) * np.linalg.norm(d[method])
        deltas[layer] = np.stack([d[name] for name in EDIT_NAMES])
    return deltas, np.rad2deg(random_angle)


def edit_protein(model, fits, layers, edit_max_tokens, sequence, ns, positions, faces):
    """All edits for the sampled residues of one protein; the masked copies of the protein run in one batch.
    layers: in ascending order (the unedited stream runs through them one after the other)."""
    assert list(layers) == sorted(layers), layers
    aa_ids = model.ids(AA)
    input_ids, _ = model.token_ids([sequence])
    n_edits = len(EDIT_NAMES) - 1  # without "none"
    per_chunk = max(1, edit_max_tokens // (n_edits * input_ids.shape[1]))
    blocks = []
    for c in range(0, len(ns), per_chunk):
        n, face = ns[c:c + per_chunk], faces[c:c + per_chunk]
        position = torch.tensor(positions[c:c + per_chunk], device=model.device) + model.offset
        rows = torch.arange(len(n), device=model.device)
        original = input_ids[0, position]
        masked = input_ids.repeat(len(n), 1)
        masked[rows, position] = model.mask_id
        vectors = [edit_vectors(fits, layers, *residue, model.hidden_size) for residue in zip(n, face)]

        # the unedited stream, kept at each ablation layer, to the last layer: the "none" edit
        h, previous, at_layer = model.embed(masked), 0, {}
        for layer in layers:
            h = model.run(h, previous, layer)
            at_layer[layer], previous = h, layer
        h = model.run(h, previous, model.n_layers)
        unedited = model.head(h[rows, position]).log_softmax(-1)  # (residues, vocabulary)

        for layer in layers:
            edited = at_layer[layer].repeat_interleave(n_edits, 0)  # row r * edits + e: residue r, edit e
            edited_rows, edited_position = torch.arange(len(edited), device=model.device), position.repeat_interleave(n_edits)
            deltas = np.concatenate([d[layer][1:] for d, _ in vectors])  # without "none"
            edited[edited_rows, edited_position] += torch.tensor(deltas, dtype=edited.dtype, device=model.device)
            edited = model.run(edited, layer, model.n_layers)
            log_probs = model.head(edited[edited_rows, edited_position]).log_softmax(-1).view(len(n), n_edits, -1)
            log_probs = torch.cat([unedited[:, None], log_probs], 1).flatten(0, 1)  # rows in the order of EDIT_NAMES
            all_rows = torch.arange(len(log_probs), device=model.device)
            blocks.append(pl.DataFrame({
                "n": np.repeat(n, len(EDIT_NAMES)), "layer": layer, "edit": EDIT_NAMES * len(n),
                "random_angle": np.repeat([angle for _, angle in vectors], len(EDIT_NAMES)),
                "edit_norm": np.linalg.norm(np.concatenate([d[layer] for d, _ in vectors]), axis=1),
                "log_likelihood": log_probs[all_rows, original.repeat_interleave(len(EDIT_NAMES))].float().cpu().numpy(),
                "top1_id": log_probs.argmax(-1).cpu().numpy(),
                "aa_log_probs": log_probs[:, aa_ids].float().cpu().numpy(),  # log-probabilities of the 20 amino acids, in the order of AA
            }))
    return pl.concat(blocks)


RNG_SCHEME = ("random angle: np.random.default_rng([0, n]).uniform(-pi, pi); random direction of a layer and method: "
              "np.random.default_rng([0, n, layer, k]).standard_normal(hidden), k = index of the method in METHODS; "
              "n = row of the residue in the sample")


def shard_path(directory, shard, layer):
    return directory / "results" / f"shard-{shard:05d}-L{layer}.parquet"


def check_results(directory, pending):
    """Stop if a file of an older run (more than one layer, or no layer in the name) has rows of a layer that is not done."""
    for path in (directory / "results").glob("shard-*.parquet"):
        if re.fullmatch(r"shard-\d+-L\d+\.parquet", path.name):
            continue  # one layer per file: a missing (shard, layer) file is computed, a present one is skipped
        found = set(pl.read_parquet(path, columns=["layer"])["layer"].unique().to_list()) & set(pending)
        if found:
            raise RuntimeError(f"{path} has rows of layers {sorted(found)}, which are not in metadata.json: "
                               "remove the file, or put these layers in metadata.json if they are complete")


def run(config, model, residues, out):
    """Edits for every residue of the sample (accession, i, helix_face, sequence), saved in shards per layer."""
    directory = out / "ablation"
    (directory / "results").mkdir(parents=True, exist_ok=True)
    metadata = directory / "metadata.json"
    previous = json.loads(metadata.read_text()) if metadata.exists() else {"layers": []}
    done = previous["layers"]
    layers = sorted(layer for layer in config["ablation_layers"] if layer not in done)  # the stream runs through the layers in order
    if not layers:
        print(f"ablation: layers {done} already done", flush=True)
        return
    print(f"ablation: layers {layers} (already done: {done})", flush=True)
    check_results(directory, layers)
    fits = load_fits(out, layers)
    vocabulary = np.array(model.vocabulary)
    numbered = residues.with_row_index("n")
    proteins = numbered.group_by("accession", maintain_order=True).agg("n", "i", "helix_face", pl.col("sequence").first())
    n_shards = (len(proteins) + SHARD_PROTEINS - 1) // SHARD_PROTEINS
    for shard in tqdm(range(n_shards), desc="ablation shards"):
        missing = [layer for layer in layers if not shard_path(directory, shard, layer).exists()]
        if not missing:
            continue
        rows = pl.concat([edit_protein(model, fits, missing, config["edit_max_tokens"], sequence, ns, positions, faces)
                          for _, ns, positions, faces, sequence in proteins[shard * SHARD_PROTEINS:(shard + 1) * SHARD_PROTEINS].iter_rows()])
        rows = (
            rows.join(numbered, on="n", maintain_order="left")
            .with_columns(
                sequence_length=pl.col("sequence").str.len_chars(), aa=pl.col("sequence").str.slice(pl.col("i"), 1),
                method=pl.col("edit").str.split(" | ").list.first(),
                control=pl.when(pl.col("edit") == "none").then(pl.lit("none"))
                         .otherwise(pl.col("edit").str.split(" | ").list.get(1, null_on_oob=True).fill_null("own angle")),
                top1_aa=pl.Series(vocabulary[rows["top1_id"].to_numpy()]),
            )
            .with_columns(
                angle_used=pl.when(pl.col("control") == "own angle").then(pl.col("helix_face"))
                            .when(pl.col("control") == "random angle").then(pl.col("random_angle")),
                top1_correct=pl.col("top1_aa") == pl.col("aa"),
            )
            .select("accession", "i", "sequence_length", "aa", "helix_face", "layer", "edit", "method", "control", "angle_used",
                    "edit_norm", "log_likelihood", "top1_correct", "top1_aa", "aa_log_probs")
        )
        for layer in missing:
            path = shard_path(directory, shard, layer)
            rows.filter(pl.col("layer") == layer).write_parquet(path.with_suffix(".tmp"))
            path.with_suffix(".tmp").rename(path)  # a shard file exists only when it is complete
    rng = previous.get("rng_per_layer", {}) | {str(layer): RNG_SCHEME for layer in layers}
    metadata.with_suffix(".tmp").write_text(json.dumps({
        "model": config["hf_id"], "layers": sorted(done + layers), "n_residues": len(residues),
        "rng_per_layer": rng, "aa_log_probs_order": AA,
    }, indent=1))
    metadata.with_suffix(".tmp").rename(metadata)  # metadata.json is always complete


def summary(out, n_boot, seed=0):
    """Mean change of log p of the correct residue and of top-1 accuracy per layer and edit, with protein-bootstrap 95% CIs."""
    results = pl.read_parquet(out / "ablation" / "results" / "*.parquet", columns=["accession", "i", "layer", "edit", "log_likelihood", "top1_correct"])
    unedited = results.filter(pl.col("edit") == "none").select("accession", "i", "layer", ll_unedited="log_likelihood", top1_unedited="top1_correct")
    effects = results.join(unedited, on=["accession", "i", "layer"]).with_columns(
        delta_ll=pl.col("log_likelihood") - pl.col("ll_unedited"),
        delta_top1=pl.col("top1_correct").cast(pl.Float64) - pl.col("top1_unedited").cast(pl.Float64),
    )
    proteins = effects["accession"].unique().sort()
    boot_weights = np.random.default_rng(seed).multinomial(len(proteins), np.full(len(proteins), 1 / len(proteins)), size=n_boot).astype(np.float64)

    def bootstrap_ci(group, column):
        per_protein = pl.DataFrame({"accession": proteins}).join(
            group.group_by("accession").agg(s=pl.col(column).sum(), n=pl.len()), on="accession", how="left").fill_null(0)
        means = (boot_weights @ per_protein["s"].to_numpy()) / (boot_weights @ per_protein["n"].to_numpy())
        return np.percentile(means, [2.5, 97.5])

    rows = []
    for (layer, name), group in effects.group_by("layer", "edit", maintain_order=True):
        ll_low, ll_high = bootstrap_ci(group, "delta_ll")
        top1_low, top1_high = bootstrap_ci(group, "delta_top1")
        rows.append({"layer": layer, "edit": name, "mean_ll_correct": group["log_likelihood"].mean(),
                     "mean_delta_ll": group["delta_ll"].mean(), "ll_ci_low": ll_low, "ll_ci_high": ll_high,
                     "top1_accuracy": group["top1_correct"].mean(), "mean_delta_top1": group["delta_top1"].mean(),
                     "top1_ci_low": top1_low, "top1_ci_high": top1_high,
                     "arith_mean_p_correct": group["log_likelihood"].exp().mean()})
    result = pl.DataFrame(rows).sort("layer", "edit")
    result.write_parquet(out / "ablation" / "summary.parquet")
    return result
