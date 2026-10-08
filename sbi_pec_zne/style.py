"""Plot style shared by the PEC, ZNE and combined figures."""

VIZ_RC = {
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "mathtext.fontset": "dejavusans",
}

# Parameter summaries in draw / legend order (truth → median → mean).
PARAM_ORDER = ["truth", "median", "mean"]
LABEL = {"mean": "Mean", "median": "Median", "truth": "Truth"}
FILL = {"truth": "#E8A09A", "median": "#7EC4BA", "mean": "#7BAFD4"}        # bar and marker fills
EDGE = {"truth": "#A63D35", "median": "#2F6B54", "mean": "#3D6D95"}        # marker edges
MARKER = {"mean": "o", "median": "s", "truth": "D"}
_MARKERSIZE = {"truth": 8.0, "median": 9.0, "mean": 9.0}
_PEC_MARKER_SCALE = 1.22    # PEC markers are slightly larger than the base size

IDEAL_STYLE = dict(color="#2C3E50", linestyle="--", linewidth=2, alpha=0.9)
NOISY_STYLE = dict(color="#8B3A62", linestyle="-.", linewidth=2, alpha=0.82)


def fill(name: str) -> str:
    return FILL.get(name, "#888888")


def markersize(name: str, default: float = 9.0) -> float:
    """Line2D marker size (pt) for the PEC / ZNE curves and legend glyphs."""
    return _MARKERSIZE.get(name, default) * _PEC_MARKER_SCALE


def scatter_area(name: str, default: float = 9.0) -> float:
    """Scatter marker area (pt²) matching the Line2D marker size of the same parameter."""
    return 1.52 * _MARKERSIZE.get(name, default) ** 2


def ordered_names(names) -> list:
    """``names`` sorted truth → median → mean, any other keys last."""
    return [n for n in PARAM_ORDER if n in names] + [n for n in names if n not in PARAM_ORDER]
