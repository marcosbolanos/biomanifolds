"""Matplotlib theme of the project: Nunito fonts, with a font weight per type of text, and the colour maps.

    from biomanifolds import theme
    theme.use()

Fonts: static instances (weights 200 to 700) of the variable Nunito font of Google Fonts, in fonts/ (SIL Open
Font License, fonts/OFL.txt).
Colour maps: HYDROPHOBICITY (red: hydrophobic, blue: hydrophilic) for values of hydrophobicity, and DELTA
(yellow: increase, purple: decrease) for changes; centre both on a meaningful value (e.g. 0) with a symmetric norm.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

FONT_DIR = Path(__file__).parent / "fonts"

# font weight per type of text, from bold to extra light
WEIGHTS = {
    "title": 200,  # extra light, very big: figure and axes titles
    "label": 500,  # medium: axis and colour bar labels
    "text": 300,  # light: tick labels, legends, other text
    "note": 200,  # extra light: small annotations, e.g. counts and values next to bars
    "highlight": 700,  # bold: a highlighted tick label or word
}

HYDROPHOBICITY = plt.cm.RdBu_r
DELTA = LinearSegmentedColormap.from_list("delta", ["#3f007d", "#8c6bb1", "#f7f7f7", "#f2d146", "#a67c00"])


def use():
    """Register the Nunito fonts and set the matplotlib defaults; in Jupyter, show figures at twice the resolution."""
    for path in sorted(FONT_DIR.glob("*.ttf")):
        font_manager.fontManager.addfont(path)
    plt.rcParams.update({
        "font.family": ["Nunito", "DejaVu Sans"],  # DejaVu Sans for the characters that Nunito does not have (e.g. θ, →)
        "font.size": 13,
        "font.weight": WEIGHTS["text"],
        "figure.titlesize": 24,
        "figure.titleweight": WEIGHTS["title"],
        "axes.titlesize": 18,
        "axes.titleweight": WEIGHTS["title"],
        "axes.labelsize": 14,
        "axes.labelweight": WEIGHTS["label"],
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "mathtext.fontset": "custom",  # formulas in Nunito too
        "mathtext.rm": "Nunito",
        "mathtext.it": "Nunito:italic",
        "mathtext.bf": "Nunito:bold",
    })
    # the fallback fonts have no light weight: matplotlib uses their regular weight, without warnings
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    # the titles are big: wrap long titles at the edges of the figure, unless a call sets wrap itself
    for cls, name in [(Figure, "suptitle"), (Axes, "set_title")]:
        method = getattr(cls, name)
        if not getattr(method, "wraps_titles", False):
            def wrapped(self, *args, _method=method, **kwargs):
                kwargs.setdefault("wrap", True)
                return _method(self, *args, **kwargs)
            wrapped.wraps_titles = True
            setattr(cls, name, wrapped)
    try:
        from matplotlib_inline.backend_inline import set_matplotlib_formats
        set_matplotlib_formats("retina")
    except ImportError:  # not in Jupyter
        pass


def inner_border(ax, radius=1):
    """Draw the inner border of a ring on polar axes, with the same line as the outer border."""
    spine = ax.spines["polar"]
    theta = np.linspace(0, 2 * np.pi, 721)
    ax.plot(theta, np.full_like(theta, radius), color=spine.get_edgecolor(), linewidth=spine.get_linewidth())
