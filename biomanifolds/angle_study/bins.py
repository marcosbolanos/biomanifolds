"""Mean residual stream per label bin (notebook 2), PCA explained variance, and the linear and harmonic fits of notebook 3.

Bins of width 1: helix face (-180, 180] degrees -> bins 0..359 (bin k: [k - 180, k - 179)), pLDDT [0, 100] -> bins 0..100.
The sums and counts of the residual stream per bin are computed in the forward pass; no embeddings are saved.
"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from tqdm.auto import tqdm

from biomanifolds import plots
from biomanifolds.angle_study.figures import save
from biomanifolds.datasets import swissprot
from biomanifolds.labels import af2_structure_labels, sequence_labels, swissprot_labels

LABELS = {"helix_face": {"bins": 360, "first": -180, "periodic": True, "widths": [1, 2, 5, 10]},
          "plddt": {"bins": 101, "first": 0, "periodic": False, "widths": [1, 2, 5]}}


def row_bins(rows):
    """Bin of every row (accession, i) for each label, in row order; -1: no value."""
    labels = (
        af2_structure_labels.load_labels(["accession", "i", "helix_face", "plddt"])
        .join(rows.with_row_index("row").lazy(), on=["accession", "i"])  # the label table is streamed on the left, only the rows are held
        .collect(engine="streaming")
    )
    row_labels = pl.DataFrame({"row": np.arange(len(rows), dtype=np.uint32)}).join(labels, on="row", how="left").sort("row")
    return {
        "helix_face": ((row_labels["helix_face"].floor() + 180) % 360).fill_null(-1).cast(pl.Int64).to_numpy(),
        "plddt": row_labels["plddt"].floor().fill_null(-1).cast(pl.Int64).to_numpy(),
    }


def accumulate(model, proteins, layers, max_tokens):
    """Sums and counts of the residual stream per bin, {(label, layer): (sums (bins, hidden), counts (bins,))},
    for the residues in `positions` of every protein (accession, sequence, positions)."""
    rows = proteins.select("accession", i="positions").explode("i")
    bins = row_bins(rows)
    starts = np.r_[0, np.cumsum(proteins["positions"].list.len().to_numpy().astype(np.int64))]  # rows of protein j: starts[j] .. starts[j + 1]
    sums = {(label, layer): torch.zeros(spec["bins"], model.hidden_size, dtype=torch.float64, device=model.device)
            for label, spec in LABELS.items() for layer in layers}
    counts = {key: np.zeros(LABELS[key[0]]["bins"]) for key in sums}

    # batches of similar lengths (little padding) with at most max_tokens tokens, special tokens included
    sequences, positions = proteins["sequence"].to_list(), proteins["positions"].to_list()
    lengths = np.array([len(s) for s in sequences])
    order, batches, batch = np.argsort(lengths), [], []
    for j in order:
        if batch and (len(batch) + 1) * (lengths[j] + 2) > max_tokens:
            batches.append(batch)
            batch = []
        batch.append(j)
    batches.append(batch)

    for batch in tqdm(batches, desc="bins", mininterval=30):
        input_ids, attention_mask = model.token_ids([sequences[j] for j in batch])
        hidden = model.hidden_states(input_ids, attention_mask, layers)
        rows_b = np.concatenate([np.arange(starts[j], starts[j + 1]) for j in batch])
        batch_index = torch.tensor(np.concatenate([np.full(len(positions[j]), b) for b, j in enumerate(batch)]), device=model.device)
        token_index = torch.tensor(np.concatenate([np.asarray(positions[j]) + model.offset for j in batch]), device=model.device)
        for layer in layers:
            h = hidden[layer][batch_index, token_index].double()  # (rows of the batch, hidden)
            for label in LABELS:
                b = bins[label][rows_b]
                keep = b >= 0
                if keep.any():
                    sums[label, layer].index_add_(0, torch.tensor(b[keep], device=model.device), h[torch.tensor(keep, device=model.device)])
                    counts[label, layer] += np.bincount(b[keep], minlength=len(counts[label, layer]))
    return {key: (sums[key].cpu().numpy(), counts[key]) for key in sums}


def bin_means(sums, counts, first_value, width, min_count):
    """Mean per bin of `width` neighbouring 1-unit bins, for the bins with at least min_count residues: means, centres, counts."""
    firsts = np.arange(0, len(counts), width)
    bin_sums, bin_counts = np.add.reduceat(sums, firsts), np.add.reduceat(counts, firsts)
    keep = bin_counts >= min_count
    return {"means": bin_sums[keep] / bin_counts[keep, None], "centers": first_value + firsts[keep] + width / 2, "counts": bin_counts[keep]}


def pca_axes(means, reference=None):
    """PC axes of `means`: centre, first 6 components and their explained variance ratio; with `reference`, the PCs point the same way."""
    center = means.mean(0)
    _, s, vt = np.linalg.svd(means - center, full_matrices=False)
    components = vt[:6]
    if reference is not None:
        components = components * np.sign((components * reference).sum(1))[:, None]
    return {"center": center, "components": components, "explained": (s**2 / (s**2).sum())[:6]}


def harmonic_fits(sums, counts):
    """Helix face: mean per 1-degree bin, global mean, and the linear [1, theta] and harmonic [1, cos, sin] least-squares fits
    of the bin means, weighted by the residues per bin; and the weighted R^2 of the two fits (notebook 3)."""
    theta = np.deg2rad(np.arange(360) - 179.5)  # centre of each 1-degree bin, in radians
    filled = counts > 0
    means = np.zeros_like(sums)
    means[filled] = sums[filled] / counts[filled, None]
    global_mean = sums.sum(0) / counts.sum()
    weights = np.sqrt(counts)[:, None]
    fits = {"bin_means": means, "global_mean": global_mean}
    for name, X in [("linear", np.column_stack([np.ones(360), theta])), ("harmonic", np.column_stack([np.ones(360), np.cos(theta), np.sin(theta)]))]:
        fits[name] = np.linalg.lstsq(X * weights, means * weights, rcond=None)[0]
    total = (counts[:, None] * (means - global_mean) ** 2).sum()
    r2 = {name: 1 - (counts[:, None] * (means - X @ fits[name]) ** 2).sum() / total
          for name, X in [("linear", np.column_stack([np.ones(360), theta])), ("harmonic", np.column_stack([np.ones(360), np.cos(theta), np.sin(theta)]))]}
    return fits, r2


def run(config, model, sizes, train, test, out):
    """Bins of the train proteins and of the test residues, PCA, fits, and figures. Results in out/bins.
    model: a function that returns the model; called only when bin sums are missing."""
    directory, figures = out / "bins", out / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    layers = config["layers"]

    # sums and counts per bin, saved so that they are only computed once
    data = {}
    for split, proteins in [("train", train), ("test", test)]:
        paths = {(label, layer): (directory / f"{split}_{label}_sums_L{layer}.npy", directory / f"{split}_{label}_counts_L{layer}.npy")
                 for label in LABELS for layer in layers}
        if not all(s.exists() and c.exists() for s, c in paths.values()):
            print(f"bins: {split}, {len(proteins):,} proteins, {proteins['positions'].list.len().sum():,} residues", flush=True)
            for key, (s, c) in accumulate(model(), proteins, layers, config["max_tokens"]).items():
                np.save(paths[key][0], s)
                np.save(paths[key][1], c)
        data[split] = {key: (np.load(s), np.load(c)) for key, (s, c) in paths.items()}

    # PCA of the train bin means; the test bin means on the train PCs
    evr, points = [], {}
    for label, spec in LABELS.items():
        for layer in layers:
            reference = None
            for width in spec["widths"]:
                train_points = bin_means(*data["train"][label, layer], spec["first"], width, sizes.min_count)
                axes = pca_axes(train_points["means"], reference)
                reference = axes["components"] if reference is None else reference
                evr += [{"label": label, "layer": layer, "width": width, "pc": k + 1, "explained": float(e)} for k, e in enumerate(axes["explained"])]
                if width == 1:
                    test_points = bin_means(*data["test"][label, layer], spec["first"], width, sizes.min_count)
                    points[label, layer] = {split: (p["centers"], (p["means"] - axes["center"]) @ axes["components"][:2].T)
                                            for split, p in [("train", train_points), ("test", test_points)]}
                np.savez(directory / f"train_{label}_width{width}_pca_axes_L{layer}.npz", **axes)
    evr = pl.DataFrame(evr)
    evr.write_parquet(directory / "pca_explained_variance.parquet")

    # linear and harmonic fits of the helix-face bin means (used by the ablation and the steering)
    r2 = []
    for layer in sorted(set(layers) | set(config["ablation_layers"]) | {config["steering_layer"]}):
        if ("helix_face", layer) not in data["train"]:
            raise ValueError(f"layer {layer} of ablation_layers or steering_layer is not in layers")
        fits, fit_r2 = harmonic_fits(*data["train"]["helix_face", layer])
        np.savez(directory / f"fits_L{layer}.npz", **fits)
        r2 += [{"layer": layer, "fit": name, "r2": float(value)} for name, value in fit_r2.items()]
    r2 = pl.DataFrame(r2)
    r2.write_parquet(directory / "fit_r2.parquet")
    print(r2, flush=True)

    figures_bins(layers, evr, points, figures)
    figure_proteins(train, test, figures)
    return evr, r2


def figures_bins(layers, evr, points, figures):
    # explained variance ratio of the first 6 PCs, per label, layer and bin width
    for label, spec in LABELS.items():
        fig, axes = plt.subplots(1, len(layers), figsize=(6 * len(layers), 4.5), sharey=True, squeeze=False)
        for ax, layer in zip(axes[0], layers):
            for k, width in enumerate(spec["widths"]):
                e = evr.filter((pl.col("label") == label) & (pl.col("layer") == layer) & (pl.col("width") == width)).sort("pc")
                ax.bar(e["pc"] + (k - (len(spec["widths"]) - 1) / 2) * 0.8 / len(spec["widths"]), e["explained"], 0.8 / len(spec["widths"]),
                       label=f"bin width {width}")
            ax.set_title(f"layer {layer}")
            ax.set_xlabel("principal component")
        axes[0][0].set_ylabel("explained variance ratio")
        axes[0][-1].legend(fontsize="small", loc="upper left", bbox_to_anchor=(1.01, 1))
        fig.suptitle(f"PCA of the train mean residual stream per {label} bin: explained variance ratio")
        fig.tight_layout()
        save(fig, figures, f"bins_{label}_pca_explained_variance")

        # the bin means on train PC1 and PC2, for train and test (bin width 1); one color scale for all panels
        if spec["periodic"]:
            vmin, vmax = -180, 180
        else:
            centers = np.concatenate([points[label, layer][split][0] for layer in layers for split in ["train", "test"]])
            vmin, vmax = centers.min(), centers.max()
        fig, axes = plt.subplots(len(layers), 2, figsize=(11, 5 * len(layers)), squeeze=False, layout="constrained")
        for row, layer in zip(axes, layers):
            for ax, split in zip(row, ["train", "test"]):
                centers, pcs = points[label, layer][split]
                order = np.r_[np.arange(len(pcs)), 0] if spec["periodic"] else np.arange(len(pcs))  # periodic: close the loop
                ax.plot(pcs[order, 0], pcs[order, 1], color="lightgray", linewidth=1, zorder=1)
                sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=centers, s=10, cmap="twilight" if spec["periodic"] else "viridis", zorder=2,
                                vmin=vmin, vmax=vmax)
                ax.set_aspect("equal", adjustable="datalim")
                ax.set_title(f"layer {layer}, {split}")
                ax.set_xlabel("train PC1")
                ax.set_ylabel("train PC2")
            fig.colorbar(sc, ax=row, shrink=0.8, label=label)
        fig.suptitle(f"Mean residual stream per {label} bin (width 1) on the train PCs", wrap=False)
        save(fig, figures, f"bins_{label}_pc1_pc2")


def figure_proteins(train, test, figures):
    """Sampled proteins against all proteins of each split (notebook 2)."""
    all_proteins = (
        sequence_labels.load_protein_labels()
        .join(swissprot_labels.load_protein_labels().select("accession", "kingdom", "group"), on="accession")
        .collect()
    )
    splits = pl.read_parquet(swissprot.swissprot_dir() / "splits" / "splits.parquet", columns=["accession", "split"])
    groups = {}
    for split, sample in [("train", train), ("test", test)]:
        everything = all_proteins.join(splits.filter(pl.col("split") == split).select("accession"), on="accession")  # all proteins of the split
        groups[f"{split} (all)"] = everything
        groups[f"{split} (sampled)"] = everything.filter(pl.col("accession").is_in(sample["accession"].implode()))
    fig = plots.protein_summary(groups)
    save(fig, figures, "bins_sampled_proteins")
