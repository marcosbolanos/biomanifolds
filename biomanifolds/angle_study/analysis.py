"""Statistics and figures of the ablation (notebook 4). Tables in out/analysis, figures in out/figures (prefix "ablation_").

95% confidence intervals from the protein bootstrap: per bootstrap sample, every protein gets a multinomial weight.
"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import rankdata

from biomanifolds import theme
from biomanifolds.angle_study.constants import AA, CONTROLS, EDITS, EISENBERG, HYDROPHOBIC, METHODS
from biomanifolds.angle_study.figures import grouped_bars, legend_right, save, wheel
from biomanifolds.angle_study.samples import sequences
from biomanifolds.labels import af2_structure_labels

PLOT_EDITS = ["bin_means", "harmonic", "bin_means | random angle"]
R_EDGES = np.linspace(-1, 1, 11)  # bins of the per-helix Pearson r
R_BINS = [f"[{low:.1f}, {high:.1f})" for low, high in zip(R_EDGES[:-1], R_EDGES[1:])]
ABS_R_EDGES = np.linspace(0, 1, 11)  # bins of |r|
ABS_R_BINS = [f"[{low:.1f}, {high:.1f})" for low, high in zip(ABS_R_EDGES[:-1], ABS_R_EDGES[1:])]
COARSE_EDGES = {2: [0.0], 4: [-0.5, 0.0, 0.5]}  # inner edges of the coarse bins of r
COARSE_BINS = {2: ["[-1, 0)", "[0, 1]"], 4: ["[-1, -0.5)", "[-0.5, 0)", "[0, 0.5)", "[0.5, 1]"]}
DELTA_EDGES = [-0.5, -0.2, -0.1, -0.05, -0.02, 0.02, 0.05, 0.1, 0.2, 0.5]  # bins of the change of log p
DELTA_BINS = ["< -0.5"] + [f"[{low}, {high})" for low, high in zip(DELTA_EDGES[:-1], DELTA_EDGES[1:])] + [">= 0.5"]
FINE_EDGES = np.linspace(-1, 1, 41)  # bins of r of width 0.05
WINDOW = 10  # residues on each side of the masked residue, in the tables of examples
R_LABEL = "Pearson r of the helix, H vs cos(θ)"


class Bootstrap:
    """Protein-bootstrap means: the same weights for every statistic, so that differences are paired."""

    def __init__(self, accessions, n_boot, seed):
        self.proteins = accessions.unique().sort()
        self.weights = np.random.default_rng(seed).multinomial(len(self.proteins), np.full(len(self.proteins), 1 / len(self.proteins)),
                                                                size=n_boot).astype(np.float64)

    def mean(self, frame, column):
        """Mean of `column` over the rows, and its bootstrap values."""
        per_protein = pl.DataFrame({"accession": self.proteins}).join(
            frame.group_by("accession").agg(s=pl.col(column).sum(), n=pl.len()), on="accession", how="left").fill_null(0)
        s, n = per_protein["s"].to_numpy(), per_protein["n"].to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            return s.sum() / n.sum(), (self.weights @ s) / (self.weights @ n)


def ci(value, boot):
    return value, np.nanpercentile(boot, 2.5), np.nanpercentile(boot, 97.5)


def run(config, out, sizes):
    directory, figures = out / "analysis", out / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    layers = config["ablation_layers"]
    results = pl.read_parquet(out / "ablation" / "results" / "*.parquet").rechunk()  # one chunk per column: to_numpy() gives views
    summary = pl.read_parquet(out / "ablation" / "summary.parquet")
    boot = Bootstrap(results["accession"], sizes.n_boot, seed=1)

    figure_methods(summary, layers, figures)
    effects = figure_distribution_and_flips(results, layers, directory, figures)
    del effects
    face_agreement(results, layers, boot, directory, figures)
    face_correlation(results, layers, sizes.n_boot, directory)
    helix_r_figures(results, layers, boot, directory, figures)
    wheels(results, layers, directory, figures)


def figure_methods(summary, layers, figures):
    """Mean change of log p of the correct residue per method and control (notebook 4, cell 2)."""
    fig, axes = plt.subplots(1, len(layers), figsize=(7 * len(layers), 4.5), sharey=True, squeeze=False)
    for ax, layer in zip(axes[0], layers):
        values, errors = {}, {}
        for control in CONTROLS:
            name = control.removeprefix(" | ") or "own angle"
            rows = [summary.filter((pl.col("layer") == layer) & (pl.col("edit") == method + control)).row(0, named=True) for method in METHODS]
            values[name] = [r["mean_delta_ll"] for r in rows]
            errors[name] = [(r["mean_delta_ll"] - r["ll_ci_low"], r["ll_ci_high"] - r["mean_delta_ll"]) for r in rows]
        grouped_bars(ax, METHODS, list(values), values, errors)
        ax.set_title(f"layer {layer}")
    axes[0][0].set_ylabel("mean Δ log p (correct residue)")
    legend_right(axes[0][-1])
    fig.suptitle("Ablation of the helix face angle: change of log p of the correct residue per method and control (95% CI), test set")
    fig.tight_layout()
    save(fig, figures, "ablation_methods")


def figure_distribution_and_flips(results, layers, directory, figures):
    """Distribution of the change of log p (bin means), and the largest drops that turned a correct top-1 into a wrong one."""
    unedited = results.filter(pl.col("edit") == "none").select("accession", "i", "layer", ll_unedited="log_likelihood", top1_unedited="top1_aa")
    effects = (
        results.filter(pl.col("edit") == "bin_means").select("accession", "i", "aa", "helix_face", "layer", "log_likelihood", "top1_aa")
        .join(unedited, on=["accession", "i", "layer"])
        .with_columns(delta_ll=pl.col("log_likelihood") - pl.col("ll_unedited"))
    )
    fig, axes = plt.subplots(1, len(layers), figsize=(7.5 * len(layers), 4), sharey=True, squeeze=False)
    for ax, layer in zip(axes[0], layers):
        ax.hist(effects.filter(pl.col("layer") == layer)["delta_ll"], bins=100)
        ax.set_yscale("log")
        ax.set_title(f"layer {layer}")
        ax.set_xlabel("Δ log p")
    axes[0][0].set_ylabel("residues")
    fig.suptitle("Distribution of the change of log p of the correct residue, bin-means ablation, test set")
    fig.tight_layout()
    save(fig, figures, "ablation_distribution")

    def context(sequence, i):
        return f"{sequence[max(0, i - WINDOW):i]}[{sequence[i]}]{sequence[i + 1:i + 1 + WINDOW]}"

    flips = (
        effects.filter((pl.col("top1_unedited") == pl.col("aa")) & (pl.col("top1_aa") != pl.col("aa")))
        .sort("delta_ll").group_by("layer", maintain_order=True).head(20)
        .join(sequences(), on="accession", maintain_order="left")
        .with_columns(context=pl.struct("sequence", "i").map_elements(lambda r: context(r["sequence"], r["i"]), return_dtype=pl.String))
        .drop("sequence")
    )
    flips.write_csv(directory / "correct_to_wrong_examples.csv")
    counts = effects.group_by("layer").agg(
        correct_to_wrong=((pl.col("top1_unedited") == pl.col("aa")) & (pl.col("top1_aa") != pl.col("aa"))).sum(),
        wrong_to_correct=((pl.col("top1_unedited") != pl.col("aa")) & (pl.col("top1_aa") == pl.col("aa"))).sum(),
    ).sort("layer")
    counts.write_csv(directory / "top1_flips.csv")
    print(counts, flush=True)
    return effects


def face_agreement(results, layers, boot, directory, figures):
    """Binary face test (notebook 4, cell 7): P(H_b(x^dec) = F) and the probability mass on the class that agrees with the face."""
    metrics = ["decoded_residue_agrees_with_face", "probability_mass_agrees_with_face"]
    log_probs = results["aa_log_probs"].cast(pl.Array(pl.Float32, len(AA))).to_numpy()  # (rows, 20), amino acids in the order of AA
    in_hydrophobic = np.array([a in HYDROPHOBIC for a in AA])
    inner_face = (results["helix_face"].abs() <= 90).to_numpy()  # F: inner face, |theta| <= 90 degrees
    m = results.select("accession", "i", "layer", "edit", "aa", "top1_aa").with_columns(
        decoded_residue_agrees_with_face=(pl.col("top1_aa").is_in(list(HYDROPHOBIC)) == pl.Series(inner_face)).cast(pl.Float64),
        probability_mass_agrees_with_face=np.where(inner_face, np.exp(log_probs[:, in_hydrophobic]).sum(1), np.exp(log_probs[:, ~in_hydrophobic]).sum(1)),
        true_residue_agrees_with_face=(pl.col("aa").is_in(list(HYDROPHOBIC)) == pl.Series(inner_face)).cast(pl.Float64),
    )
    before = m.filter(pl.col("edit") == "none").select("accession", "i", "layer", *[pl.col(x).alias(f"{x} before") for x in metrics])
    changes = (
        m.filter(pl.col("edit").is_in(EDITS)).select("accession", "i", "layer", "edit", *metrics)
        .join(before, on=["accession", "i", "layer"])
        .with_columns(*[(pl.col(x) - pl.col(f"{x} before")).alias(f"{x} change") for x in metrics])
    )
    rows = []
    for layer in layers:
        for metric in metrics:
            unedited = m.filter((pl.col("layer") == layer) & (pl.col("edit") == "none"))
            stats = {"before": boot.mean(unedited, metric), "true residues": boot.mean(unedited, "true_residue_agrees_with_face")}
            for edit in EDITS:
                stats[f"change: {edit}"] = boot.mean(changes.filter((pl.col("layer") == layer) & (pl.col("edit") == edit)), f"{metric} change")
            for name, (value, b) in stats.items():
                rows.append({"layer": layer, "metric": metric, "statistic": name, **dict(zip(["value", "ci_low", "ci_high"], ci(value, b)))})
    table = pl.DataFrame(rows)
    table.write_parquet(directory / "face_agreement.parquet")

    layer = max(layers)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    values, errors = {}, {}
    for control in CONTROLS:
        name = control.removeprefix(" | ") or "own angle"
        r = [table.filter((pl.col("layer") == layer) & (pl.col("metric") == metrics[0]) & (pl.col("statistic") == f"change: {method}{control}"))
             .row(0, named=True) for method in METHODS]
        values[name] = [x["value"] for x in r]
        errors[name] = [(x["value"] - x["ci_low"], x["ci_high"] - x["value"]) for x in r]
    grouped_bars(ax, METHODS, list(values), values, errors)
    ax.set_ylabel(r"$\Delta P(H_b(x^{dec}) = F)$")
    ax.set_ylim(bottom=min(-0.2, ax.get_ylim()[0]))
    ax.legend(fontsize="small")
    ax.set_title(rf"Ablation: $\Delta P(H_b(x^{{dec}}) = F)$ per method and control, layer {layer} (95% CI), test set")
    fig.tight_layout()
    save(fig, figures, "ablation_decoded_residue_agrees_with_face")
    return table


def face_correlation(results, layers, n_boot, directory, seed=1):
    """Pearson, Spearman and slope of the expected hydrophobicity of the prediction against cos(face) (notebook 4, cell 9)."""
    metrics = ["pearson", "spearman", "slope"]
    boot_rng = np.random.default_rng(seed)
    h = (
        results.select("accession", "i", "layer", "edit", "aa", "helix_face")
        .with_columns(h_pred=np.exp(results["aa_log_probs"].cast(pl.Array(pl.Float32, len(AA))).to_numpy()) @ np.array([EISENBERG[a] for a in AA]))
        .filter(pl.col("aa").is_in(list(EISENBERG)))
        .sort("layer", "edit", "accession", "i")
        .with_columns(h_true=pl.col("aa").replace_strict(EISENBERG), cos_face=pl.col("helix_face").radians().cos())
    )

    def statistics(x, y):
        pearson = np.corrcoef(x, y)[0, 1]
        spearman = np.corrcoef(rankdata(x), rankdata(y))[0, 1]
        return np.array([pearson, spearman, np.cov(x, y)[0, 1] / np.var(x, ddof=1)])

    def weighted_statistics(sums, w):
        n, x, y, xx, yy, xy = (w @ s for s in sums)
        cov, var_x, var_y = xy / n - x * y / n**2, xx / n - (x / n) ** 2, yy / n - (y / n) ** 2
        return cov / np.sqrt(var_x * var_y), cov / var_x

    rows = []
    for layer in layers:
        unedited = h.filter((pl.col("layer") == layer) & (pl.col("edit") == "none"))
        x = unedited["cos_face"].to_numpy()
        series = {"before": unedited["h_pred"].to_numpy(), "true residues": unedited["h_true"].to_numpy(),
                  **{edit: h.filter((pl.col("layer") == layer) & (pl.col("edit") == edit))["h_pred"].to_numpy() for edit in EDITS}}
        # Pearson and slope from per-protein sums; Spearman: Pearson r of the ranks of the full sample
        proteins, protein_of_residue = np.unique(unedited["accession"].to_numpy(), return_inverse=True)
        weights = boot_rng.multinomial(len(proteins), np.full(len(proteins), 1 / len(proteins)), size=n_boot).astype(np.float64)

        def per_protein_sums(a, b):
            return [np.bincount(protein_of_residue, v, len(proteins)) for v in [np.ones_like(a), a, b, a * a, b * b, a * b]]

        boot = {}
        for name, y in series.items():
            pearson, slope = weighted_statistics(per_protein_sums(x, y), weights)
            spearman, _ = weighted_statistics(per_protein_sums(rankdata(x), rankdata(y)), weights)
            boot[name] = np.column_stack([pearson, spearman, slope])
        estimate = {name: statistics(x, y) for name, y in series.items()}
        for k, metric in enumerate(metrics):
            for name in ["before", "true residues"]:
                rows.append({"layer": layer, "metric": metric, "statistic": name, "value": estimate[name][k],
                             "ci_low": np.percentile(boot[name][:, k], 2.5), "ci_high": np.percentile(boot[name][:, k], 97.5)})
            for edit in EDITS:
                change = boot[edit][:, k] - boot["before"][:, k]
                rows.append({"layer": layer, "metric": metric, "statistic": f"change: {edit}", "value": estimate[edit][k] - estimate["before"][k],
                             "ci_low": np.percentile(change, 2.5), "ci_high": np.percentile(change, 97.5)})
    table = pl.DataFrame(rows)
    table.write_parquet(directory / "face_correlation.parquet")
    return table


def helix_r_figures(results, layers, boot, directory, figures):
    """Effects by the helix's correlation r between hydrophobicity and cos(face) (notebook 4, cell 11)."""
    helix_residues = (
        af2_structure_labels.load_labels(["accession", "i", "helix_face", "pos_in_seg"])
        .filter(pl.col("helix_face").is_not_null())
        .join(results.select("accession").unique().lazy(), on="accession")
        .join(sequences().lazy(), on="accession")
        .with_columns(helix_start=pl.col("i") - pl.col("pos_in_seg"), aa=pl.col("sequence").str.slice(pl.col("i"), 1))
        .drop("sequence")
        .filter(pl.col("aa").is_in(list(EISENBERG)))
        .collect(engine="streaming")
    )
    helix_r = helix_residues.group_by("accession", "helix_start").agg(
        helix_r=pl.corr(pl.col("aa").replace_strict(EISENBERG, return_dtype=pl.Float64), pl.col("helix_face").radians().cos()))
    per_residue = (
        results.filter(pl.col("edit").is_in(["none", *PLOT_EDITS])).select("accession", "i", "layer", "edit", "log_likelihood", "top1_correct")
        .join(helix_residues.select("accession", "i", "helix_start"), on=["accession", "i"])
        .join(helix_r, on=["accession", "helix_start"])
        .join(results.filter(pl.col("edit") == "none").select("accession", "i", "layer", ll_unedited="log_likelihood", top1_unedited="top1_correct"),
              on=["accession", "i", "layer"])
        .with_columns(delta_ll=pl.col("log_likelihood") - pl.col("ll_unedited"),
                      correct=pl.col("top1_correct").cast(pl.Float64) * 100,
                      delta_correct=(pl.col("top1_correct").cast(pl.Float64) - pl.col("top1_unedited").cast(pl.Float64)) * 100,
                      correct_to_wrong=(~pl.col("top1_correct")).cast(pl.Float64) * 100,
                      r_bin=pl.col("helix_r").bin_intervals(R_EDGES[1:-1].tolist(), labels=R_BINS).cast(pl.String))
        .with_columns(delta_bin=pl.col("delta_ll").bin_intervals(DELTA_EDGES, labels=DELTA_BINS).cast(pl.String),
                      abs_helix_r=pl.col("helix_r").abs(),
                      coarse2_bin=pl.col("helix_r").bin_intervals(COARSE_EDGES[2], labels=COARSE_BINS[2]).cast(pl.String),
                      coarse4_bin=pl.col("helix_r").bin_intervals(COARSE_EDGES[4], labels=COARSE_BINS[4]).cast(pl.String),
                      abs_r_bin=pl.col("helix_r").abs().bin_intervals(ABS_R_EDGES[1:-1].tolist(), labels=ABS_R_BINS).cast(pl.String))
        .filter(pl.col("helix_r").is_not_nan())  # helices of a single amino-acid class value have no r
    )
    edits = [(edit, edit) for edit in PLOT_EDITS]

    def bar_plot(name, column, series, ylabel, title, bin_column="r_bin", bins=R_BINS, xlabel=R_LABEL, data=per_residue):
        fig, axes = plt.subplots(1, len(layers), figsize=(9 * len(layers), 4.5), sharey=True, squeeze=False)
        for ax, layer in zip(axes[0], layers):
            counts = data.filter((pl.col("layer") == layer) & (pl.col("edit") == series[0][1]))[bin_column].value_counts()
            counts = dict(zip(counts[bin_column], counts["count"]))
            values, errors = {}, {}
            for label, edit in series:
                values[label], errors[label] = [], []
                for b in bins:
                    frame = data.filter((pl.col("layer") == layer) & (pl.col("edit") == edit) & (pl.col(bin_column) == b))
                    if len(frame) == 0:
                        values[label].append(np.nan), errors[label].append((0, 0))
                        continue
                    value, low, high = ci(*boot.mean(frame, column))
                    values[label].append(value), errors[label].append((value - low, high - value))
            grouped_bars(ax, bins, [label for label, _ in series], values, errors, [f"{b}\nn={counts.get(b, 0)}" for b in bins])
            ax.tick_params(axis="x", labelrotation=45, labelsize="x-small")
            ax.set_title(f"layer {layer}")
            ax.set_xlabel(xlabel)
        axes[0][0].set_ylabel(ylabel)
        legend_right(axes[0][-1])
        fig.suptitle(title)
        fig.tight_layout()
        save(fig, figures, name)

    def line_plot(name, column, series, ylabel, title, data=per_residue):
        groups = data.select("accession", "layer", "edit", "helix_r", column).with_columns(
            fine_bin=pl.col("helix_r").bin_intervals(FINE_EDGES[1:-1].tolist(), labels=False)).partition_by("layer", "edit", "fine_bin", as_dict=True)
        centres = (FINE_EDGES[:-1] + FINE_EDGES[1:]) / 2
        fig, axes = plt.subplots(1, len(layers), figsize=(9 * len(layers), 4.5), sharey=True, squeeze=False)
        for ax, layer in zip(axes[0], layers):
            for label, edit in series:
                stats = [ci(*boot.mean(groups[layer, edit, b], column)) if (layer, edit, b) in groups else (np.nan,) * 3 for b in range(len(centres))]
                value, low, high = np.array(stats).T
                ax.plot(centres, value, marker=".", label=label)
                ax.fill_between(centres, low, high, alpha=0.3)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_title(f"layer {layer}")
            ax.set_xlabel(f"{R_LABEL} (bins of 0.05)")
        axes[0][0].set_ylabel(ylabel)
        legend_right(axes[0][-1])
        fig.suptitle(title)
        fig.tight_layout()
        save(fig, figures, name)

    by_r = "by the helix's correlation between hydrophobicity and cos(face)"
    bar_plot("ablation_delta_ll_by_r", "delta_ll", edits, "mean Δ log p (correct residue)", f"Change of log p of the correct residue, {by_r}")
    line_plot("ablation_delta_ll_by_r_fine", "delta_ll", edits, "mean Δ log p (correct residue)", f"Change of log p of the correct residue, {by_r}")
    for n_bins in [2, 4]:
        bar_plot(f"ablation_delta_top1_by_r_{n_bins}_bins", "delta_correct", edits, "Δ top-1 accuracy (percentage points)",
                 f"Change of top-1 accuracy after the edit, {by_r} ({n_bins} bins)", bin_column=f"coarse{n_bins}_bin", bins=COARSE_BINS[n_bins])
    line_plot("ablation_top1_by_r_fine", "correct", [("unedited", "none")] + edits, "% correct top-1", f"Residues predicted correctly (top-1), {by_r}")
    line_plot("ablation_delta_top1_by_r_fine", "delta_correct", edits, "Δ top-1 accuracy (percentage points)", f"Change of top-1 accuracy after the edit, {by_r}")
    correct_before = per_residue.filter(pl.col("top1_unedited"))
    bar_plot("ablation_correct_to_wrong_by_r", "correct_to_wrong", edits, "% wrong after the edit",
             f"Residues predicted correctly before the edit and wrongly after it, {by_r}", data=correct_before)
    line_plot("ablation_correct_to_wrong_by_r_fine", "correct_to_wrong", edits, "% wrong after the edit",
              f"Residues predicted correctly before the edit and wrongly after it, {by_r}", data=correct_before)
    bar_plot("ablation_delta_ll_by_abs_r", "delta_ll", edits, "mean Δ log p (correct residue)",
             "Change of log p of the correct residue, by the strength |r| of the helix correlation (either sign)",
             bin_column="abs_r_bin", bins=ABS_R_BINS, xlabel="|Pearson r| of the helix, H vs cos(θ)")
    bar_plot("ablation_log_p_by_r", "log_likelihood", [("unedited", "none")] + edits, "mean log p (correct residue)", f"Log p of the correct residue, {by_r}")

    # distribution of the per-helix r: over the helices, and over the steered residues
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (name, values) in zip(axes, [("helices", helix_r["helix_r"].fill_nan(None).drop_nulls()),
                                         ("steered residues", per_residue.filter((pl.col("layer") == layers[0]) & (pl.col("edit") == "none"))["helix_r"])]):
        ax.hist(values, bins=np.linspace(-1, 1, 41))
        ax.axvline(values.mean(), color="black", linestyle="--", label=f"mean {values.mean():.3f}, median {values.median():.3f}")
        ax.set_title(f"{name} ({len(values):,})")
        ax.set_xlabel(R_LABEL)
        ax.legend(fontsize="small")
    axes[0].set_ylabel("count")
    fig.tight_layout()
    save(fig, figures, "ablation_helix_r_distribution")

    by_delta = "by the change of log p of the correct residue"
    bar_plot("ablation_r_by_delta_ll", "helix_r", edits, "mean Pearson r of the helix", f"Helix correlation between hydrophobicity and cos(face), {by_delta}",
             bin_column="delta_bin", bins=DELTA_BINS, xlabel="Δ log p of the correct residue (n: residues for bin_means)")
    bar_plot("ablation_abs_r_by_delta_ll", "abs_helix_r", edits, "mean |Pearson r| of the helix", f"Strength of the helix correlation |r| (either sign), {by_delta}",
             bin_column="delta_bin", bins=DELTA_BINS, xlabel="Δ log p of the correct residue (n: residues for bin_means)")

    # mean Eisenberg hydrophobicity of the true residues per helix face angle (1-degree bins)
    by_angle = (
        helix_residues.with_columns(angle_bin=((pl.col("helix_face") + 180) // 1).clip(0, 359).cast(pl.Int32),
                                    h=pl.col("aa").replace_strict(EISENBERG, return_dtype=pl.Float64))
        .group_by("angle_bin").agg(mean_h=pl.col("h").mean(), n=pl.len()).sort("angle_bin")
    )
    by_angle.write_parquet(directory / "true_hydrophobicity_by_angle.parquet")
    mean_h = by_angle["mean_h"].to_numpy()
    norm = Normalize(-np.abs(mean_h).max(), np.abs(mean_h).max())  # centred on 0: the sign separates hydrophobic and hydrophilic
    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw={"projection": "polar"}, layout="constrained")
    wheel(ax, by_angle["angle_bin"].to_numpy() - 180 + 0.5, mean_h, 1, norm, theme.HYDROPHOBICITY, "")
    fig.colorbar(ScalarMappable(norm, theme.HYDROPHOBICITY), ax=ax, shrink=0.7, label="mean Eisenberg hydrophobicity (consensus scale)")
    fig.suptitle(f"Mean Eisenberg hydrophobicity per helix face angle\n(1° bins, {by_angle['n'].sum():,} helix residues), test set", wrap=False)
    save(fig, figures, "true_hydrophobicity_by_angle")


def wheels(results, layers, directory, figures, width=5):
    """Hydrophobicity of the prediction per helix face angle, before and after the harmonic ablation (notebook 4, cell 13)."""
    scores = {"hydrophobic_mass": ("Hydrophobic probability mass (Eisenberg > 0)", "mean Σ p(aa), aa hydrophobic (probability)"),
              "h_pred": ("Expected Eisenberg hydrophobicity of the prediction", "mean Σ p(aa)·H(aa) (Eisenberg consensus scale)")}
    in_hydrophobic = np.array([a in HYDROPHOBIC for a in AA])
    steered = results.filter(pl.col("edit").is_in(["none", "harmonic"])).select("accession", "i", "layer", "edit", "helix_face", "aa_log_probs")
    p = np.exp(steered["aa_log_probs"].cast(pl.Array(pl.Float32, len(AA))).to_numpy())
    per_row = steered.drop("aa_log_probs").with_columns(
        hydrophobic_mass=p[:, in_hydrophobic].sum(1),  # raw sum over the 20 amino acids, not renormalised
        h_pred=p @ np.array([EISENBERG[a] for a in AA], dtype=np.float32),
        angle_bin=((pl.col("helix_face") + 180) // width).clip(0, 360 // width - 1).cast(pl.Int32),
    )
    del p
    table = (
        per_row.filter(pl.col("edit") == "none")
        .join(per_row.filter(pl.col("edit") == "harmonic").select("accession", "i", "layer", *[pl.col(s).alias(f"{s} after") for s in scores]),
              on=["accession", "i", "layer"])
        .group_by("layer", "angle_bin")
        .agg(n=pl.len(), *[pl.col(s).mean().alias(f"{s} before") for s in scores], *[pl.col(f"{s} after").mean() for s in scores],
             *[(pl.col(f"{s} after") - pl.col(s)).mean().alias(f"{s} change") for s in scores])
        .sort("layer", "angle_bin")
    )
    table.write_parquet(directory / "prediction_hydrophobicity_by_angle.parquet")
    for score, (name, label) in scores.items():
        both = table.select(f"{score} before", f"{score} after").to_numpy()
        norm = Normalize(-np.abs(both).max(), np.abs(both).max()) if score == "h_pred" else Normalize(both.min(), both.max())
        limit = np.abs(table[f"{score} change"].to_numpy()).max()
        change_norm = Normalize(-limit, limit)
        fig, axes = plt.subplots(len(layers), 3, figsize=(17, 6 * len(layers)), subplot_kw={"projection": "polar"}, squeeze=False, layout="constrained")
        fig.get_layout_engine().set(wspace=0.02, h_pad=0.15)
        for row, layer in zip(axes, layers):
            w = table.filter(pl.col("layer") == layer)
            angles = w["angle_bin"].to_numpy() * width - 180 + width / 2
            wheel(row[0], angles, w[f"{score} before"].to_numpy(), width, norm, theme.HYDROPHOBICITY, f"layer {layer}, before (unedited)")
            wheel(row[1], angles, w[f"{score} after"].to_numpy(), width, norm, theme.HYDROPHOBICITY, f"layer {layer}, after (harmonic ablation)")
            wheel(row[2], angles, w[f"{score} change"].to_numpy(), width, change_norm, theme.DELTA, f"layer {layer}, change (after - before)")
            fig.colorbar(ScalarMappable(norm, theme.HYDROPHOBICITY), ax=row[:2], shrink=0.7, label=label)
            fig.colorbar(ScalarMappable(change_norm, theme.DELTA), ax=row[2], shrink=0.7, label=f"Δ {label}")
        fig.suptitle(f"{name} per helix face angle, before and after the harmonic ablation of the residue's own angle\n"
                     f"({width}° bins, {w['n'].sum():,} residues per layer), test set", wrap=False)
        save(fig, figures, f"ablation_wheels_{score}")
