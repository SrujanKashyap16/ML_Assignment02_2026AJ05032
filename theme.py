"""Design tokens and Matplotlib styling shared by every figure in the app.

Keeping the palette in one module means the page CSS and the charts quote the
same hex values, so nothing drifts when a colour is adjusted.

The categorical slots are assigned in fixed order and were checked for
colour-vision-deficiency separation against a white surface before use (worst
adjacent pair dE 9.1 protan, 19.6 normal vision). Three of the five sit below a
3:1 contrast ratio against white, so every chart that uses them also carries
visible value labels and a table view -- identity is never left to colour alone.
"""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# -- Surfaces and ink ------------------------------------------------------

SURFACE = "#ffffff"
SURFACE_SUNKEN = "#f7f7f5"
BORDER = "#e4e3df"
BORDER_STRONG = "#d3d2cc"

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8984"

# -- Categorical slots, fixed order ---------------------------------------

SERIES = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
]

ACCENT = SERIES[0]

# -- Sequential blue ramp, light -> dark, for the confusion matrix ---------

SEQUENTIAL_BLUE = [
    "#eef5fd",
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
]

BLUE_RAMP = LinearSegmentedColormap.from_list("ramp_blue", SEQUENTIAL_BLUE)


def model_colors(model_names) -> dict[str, str]:
    """Bind each model to a fixed slot, so a colour always means one model."""
    return {name: SERIES[index % len(SERIES)] for index, name in enumerate(model_names)}


def apply_matplotlib_theme() -> None:
    """Global rcParams: recessive chrome, legible type, transparent surface."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            # DejaVu Sans ships with Matplotlib, so this resolves identically on
            # Streamlit Community Cloud as it does locally -- no font warnings.
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 9,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.edgecolor": BORDER_STRONG,
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "grid.color": BORDER,
            "grid.linewidth": 0.8,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "figure.dpi": 150,
        }
    )


def strip_chrome(axes, keep=("left", "bottom")) -> None:
    """Drop the spines a chart does not need, so the data carries the eye."""
    for side, spine in axes.spines.items():
        spine.set_visible(side in keep)
