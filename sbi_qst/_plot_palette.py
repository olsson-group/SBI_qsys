"""Plot colours and markers of the PEC/ZNE figures (copy of ``sbi_pec_zne/style.py``, so that sbi_qst
does not import qiskit)."""

# Truth -> Median -> Mean.
COMBINED_BAR_COLORS = {
    "truth": "#E8A09A",   # pastel red
    "median": "#7EC4BA",  # pastel green
    "mean": "#7BAFD4",    # pastel blue
}

_PARAM_LINE2D_MARKERSIZE = {
    "truth": 8.0,
    "median": 9.0,
    "mean": 9.0,
}

_PEC_LINE2D_MARKERSIZE_SCALE = 1.22


def param_line2d_markersize_pec(name: str, *, default: float = 9.0) -> float:
    return float(_PARAM_LINE2D_MARKERSIZE.get(name, default)) * _PEC_LINE2D_MARKERSIZE_SCALE


_MARKER_EDGES = {
    "truth": "#A63D35",
    "median": "#2F6B54",
    "mean": "#3D6D95",
}
_MARKERS = {
    "mean": "o",
    "median": "s",
    "truth": "D",
}
