"""Figure helpers of the angle study: saving, wheels, and bars or lines with bootstrap confidence intervals."""

import matplotlib.pyplot as plt
import numpy as np

from biomanifolds import theme


def save(fig, directory, name):
    """Save a figure as <directory>/<name>.png, and close it."""
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure {name}.png", flush=True)


def wheel(ax, angles, values, width, norm, cmap, title):
    """Ring of sectors at `angles` (degrees, centres) coloured by `values`, 0 degrees (toward the protein centroid) at the top."""
    ax.bar(np.deg2rad(angles), 1, bottom=1, width=np.deg2rad(width), color=cmap(norm(values)), linewidth=0, antialiased=False)  # no borders between sectors
    ax.set_rlim(0, 2)
    ax.set_yticks([])
    ax.grid(False)
    ax.set_theta_zero_location("N")  # 0 degrees at the top
    ax.set_thetagrids([0, 90, 180, 270], ["0°", "90°", "180°", "-90°"])
    theme.inner_border(ax)
    ax.set_title(title, pad=28)  # above the 0° label


def grouped_bars(ax, groups, series, values, errors, labels=None):
    """Bars of values[series][group] with errors[series][group] = (low, high) distances, one colour per series."""
    width = 0.8 / len(series)
    for k, name in enumerate(series):
        ax.bar(np.arange(len(groups)) + (k - (len(series) - 1) / 2) * width, values[name], width,
               yerr=np.array(errors[name]).T, capsize=2, label=name)
    ax.set_xticks(np.arange(len(groups)), labels or groups)
    ax.axhline(0, color="black", linewidth=0.5)


def legend_right(ax):
    ax.legend(fontsize="small", loc="upper left", bbox_to_anchor=(1.01, 1))  # outside, to the right
