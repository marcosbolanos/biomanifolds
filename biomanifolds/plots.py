import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.transforms import blended_transform_factory

KINGDOMS = ["Bacteria", "Archaea", "Eukaryota", "Viruses", "other"]
MAJOR_GROUPS = ["Pseudomonadota", "Bacillota", "Actinomycetota", "other Bacteria", "Archaea", "Metazoa",
                "Fungi", "Viridiplantae", "other Eukaryota", "Viruses", "other"]


def _stacked_bars(ax, groups, y, column, categories, colors, title):
    left = np.zeros(len(groups))
    for category, color in zip(categories, colors):
        fractions = np.array([(df[column] == category).mean() for df in groups.values()])
        ax.barh(y, fractions, left=left, color=color, label=category)
        for yi, start, fraction in zip(y, left, fractions):
            if fraction >= 0.05:
                ax.text(start + fraction / 2, yi, f"{fraction:.0%}", ha="center", va="center", fontsize=8)
        left += fractions
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of proteins")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False, fontsize=8)


def protein_summary(groups: dict[str, pl.DataFrame], max_length=1022):
    """Sequence length, kingdom and major taxonomic group of groups of proteins.

    `groups` maps a row name to a frame with one row per protein and the columns length,
    kingdom and group."""
    labels = [f"{name}  (n={len(df):,})" for name, df in groups.items()]
    y = np.arange(len(groups))[::-1]

    fig = plt.figure(figsize=(14, 4 + 1.5 * len(groups)))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.35, wspace=0.05)

    # 1. Sequence length
    ax = fig.add_subplot(grid[0, :])
    log_lengths = [np.log10(df["length"].to_numpy()) for df in groups.values()]
    parts = ax.violinplot(log_lengths, positions=y, orientation="horizontal", showextrema=False, widths=0.9)
    for body in parts["bodies"]:
        body.set_facecolor("steelblue")
        body.set_alpha(0.5)
    for yi, values in zip(y, log_lengths):
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        ax.hlines(yi, q1, q3, color="black", lw=3)
        ax.scatter(median, yi, color="white", edgecolor="black", zorder=3)
        ax.text(1.01, yi, f"median {10**median:.0f}", va="center",
                transform=blended_transform_factory(ax.transAxes, ax.transData))
    ax.axvline(np.log10(max_length), color="firebrick", ls="--", lw=1)
    ax.text(np.log10(max_length) + 0.02, y[-1] - 0.6, f"{max_length} (ESM-2 limit)", color="firebrick")
    ticks = [10, 30, 100, 300, 1000, 3000, 10000, 30000]
    ax.set_xticks(np.log10(ticks), [str(t) for t in ticks])
    ax.set_yticks(y, labels)
    ax.set_xlabel("sequence length (residues, log scale)")
    ax.set_title("1. Sequence length")

    # 2a. Kingdom
    ax = fig.add_subplot(grid[1, 0])
    _stacked_bars(ax, groups, y, "kingdom", KINGDOMS, plt.cm.Set2([0, 2, 4, 6, 7]), "2a. Kingdom")
    ax.set_yticks(y, labels, fontsize=8)

    # 2b. Major groups
    ax = fig.add_subplot(grid[1, 1])
    _stacked_bars(ax, groups, y, "group", MAJOR_GROUPS, plt.cm.tab20(range(11)), "2b. Major groups")
    ax.set_yticks(y, [])
    return fig
