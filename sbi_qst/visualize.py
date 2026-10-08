"""Figures of the QST + NPE results: fidelity histograms, Re/Im ρ heatmaps, prep-angle sweep, simulation-budget panels."""

from __future__ import annotations

import math
import os

import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from sbi_qst._plot_palette import COMBINED_BAR_COLORS, param_line2d_markersize_pec, _MARKER_EDGES, _MARKERS
from sbi_qst.physics import (
    brickwall_rx_ry_state_vector,
    fidelity_mps_tensors,
    fidelity_pure,
    prep_to_theta,
    rho_from_psi,
    theta_to_mps_tensors,
    theta_to_psi_and_rho,
)

# The dense Re/Im ρ heatmaps are skipped when n_qubits exceeds the limit passed to
# ``plot_all`` (``max_n_qubits_dense_rho_plots``) or this module default.
_DEFAULT_MAX_N_QUBITS_DENSE_RHO_PLOTS = 99


def _boost_saturation(hex_color: str, scale: float = 1.2) -> str:
    h, s, v = mcolors.rgb_to_hsv(mcolors.to_rgb(hex_color))
    return mcolors.to_hex(mcolors.hsv_to_rgb((h, float(max(0.0, min(1.0, s * scale))), v)))


# Blue → white → muted red diverging map for the Re/Im(ρ) and sweep heatmaps.
_CMAP_MUTED_DIVERGING = mcolors.LinearSegmentedColormap.from_list(
    "muted_diverging_blue_white_red",
    [
        _boost_saturation("#67accb"),
        _boost_saturation("#67accb"),
        _boost_saturation("#8ac0d8"),
        _boost_saturation("#b3d4e5"),
        _boost_saturation("#dae9f0"),
        "#fefefe",
        _boost_saturation("#efdfe0"),
        _boost_saturation("#e0bdc1"),
        _boost_saturation("#d49ba2"),
        _boost_saturation("#c17b83"),
        _boost_saturation("#c17b83"),
    ],
    N=256,
)

_FIG_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.linewidth": 0.9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
}

# Typography of the fidelity histograms.
_QST_FIDELITY_HIST_LABEL_FONTSIZE = 12
_QST_FIDELITY_HIST_TICK_FONTSIZE = 10
_QST_FIDELITY_HIST_LEGEND_FONTSIZE = 9
_QST_FIDELITY_HIST_TICK_LENGTH = 4.0
_QST_FIDELITY_HIST_TICK_WIDTH = 0.9


def _fidelity_stat_legend_label(name: str, value: float, *, use_infidelity: bool = False) -> str:
    """Mathtext label: F_sub = value (entire legend entry in math font)."""
    symbol = "1-F" if use_infidelity else "F"
    fmt = f"{value:.2e}" if (use_infidelity and 0 < value < 1e-3) else f"{value:.4f}"
    return rf"${symbol}_{{\mathrm{{{name}}}}}={fmt}$"


def _plain_decimal_log_formatter(v: float, _pos: int = 0) -> str:
    """Format a log-axis tick (power of ten) as scientific notation, except 10^0 -> '1'."""
    if not np.isfinite(v) or v <= 0:
        return ""
    if abs(v - 1.0) < 1e-9:
        return "1"
    exp = int(round(np.log10(v)))
    return rf"$10^{{{exp}}}$"


def _apply_training_data_line_axis_style(
    ax: plt.Axes, *, panel: bool = False, legend_fontsize: float | None = None,
) -> None:
    """Match axis label / tick / legend sizes to sim-budget line plot RC."""
    rc = _PANEL_TRAINING_DATA_LINE_RC if panel else _TRAINING_DATA_LINE_RC
    ax.xaxis.label.set_fontsize(rc["axes.labelsize"])
    ax.yaxis.label.set_fontsize(rc["axes.labelsize"])
    labelpad = 1.5 if panel else 3.0
    ax.xaxis.labelpad = labelpad
    ax.yaxis.labelpad = labelpad
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=rc["xtick.major.size"],
        width=rc["xtick.major.width"],
        labelsize=rc["xtick.labelsize"],
        pad=1.5 if panel else 3.0,
    )
    leg = ax.get_legend()
    if leg is not None:
        leg_fs = legend_fontsize if legend_fontsize is not None else rc["legend.fontsize"]
        for text in leg.get_texts():
            text.set_fontsize(leg_fs)


def _align_panel_xaxis_labels(axes: list[plt.Axes], *, labelpad: float = 3.0) -> None:
    """Align x-axis label baselines across panel axes (match left / reference bottom)."""
    if not axes:
        return
    fig = axes[0].figure
    for ax in axes:
        ax.xaxis.labelpad = labelpad
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ref_bottom = axes[0].xaxis.label.get_window_extent(renderer).y0
    for ax in axes[1:]:
        lo, hi = labelpad, labelpad + 24.0
        best_pad = labelpad
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            ax.xaxis.labelpad = mid
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bottom = ax.xaxis.label.get_window_extent(renderer).y0
            if abs(bottom - ref_bottom) <= 0.5:
                best_pad = mid
                break
            if bottom > ref_bottom:
                lo = mid
            else:
                hi = mid
            best_pad = mid
        ax.xaxis.labelpad = best_pad
    fig.canvas.draw()


# Line-plot style shared with ``sbi_paulinoise/visualize.py``.
_TRAINING_DATA_LINE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "mathtext.fontset": "dejavusans",
    "font.size": 11,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}

# Larger typography of the two-panel sim-budget + histogram figure.
_PANEL_TRAINING_DATA_LINE_RC = {
    **_TRAINING_DATA_LINE_RC,
    "font.size": 13,
    "axes.labelsize": 14,
    "legend.fontsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
}

# Histogram x padding as a fraction of its span, equal to the left panel's x margin (0.30 / 8 budget units).
_PANEL_X_SIDE_PAD_FRAC = 0.30 / 8.0


def _set_rho_axes_ket_ticks(ax: plt.Axes, n_qubits: int, *, label_fontsize: float = 8) -> None:
    """Minimal ρ heatmap axes: no ticks or axis labels (colorbar carries scale)."""
    del n_qubits, label_fontsize
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)


def _rho_panel_size_in(n_qubits: int) -> float:
    """Square panel edge (inches) for dense ρ imshow.

    Small n: compact panels where individual entries can still be seen.
    Large n (e.g. 12): larger square overview panels — d=4096 is always downsampled.
    """
    n = int(n_qubits)
    if n <= 4:
        return 2.4
    if n <= 6:
        return 2.6
    if n <= 8:
        return 2.9
    return 3.25


def _rho_real_imag_figsize(n_qubits: int) -> tuple[float, float]:
    p = _rho_panel_size_in(n_qubits)
    return (2.0 * p + 0.35, 2.0 * p + 0.55)


def posterior_sample_fidelities(
    samples: np.ndarray,
    psi_true: np.ndarray | None,
    *,
    n_qubits: int,
    state_param: str,
    tensor_bond_dim: int = 2,
    prep_theta: np.ndarray | None = None,
    ansatz_layers: int = 1,
    unified_ry_rx: bool = False,
    cnot_topology: str = "linear",
) -> np.ndarray:
    r"""Per-sample fidelity :math:`F_i` of posterior samples against the true state.

    ``cholesky``: :math:`|\langle\psi_{\mathrm{true}}|\psi_i\rangle|^2` with the dense ``psi_true``.
    ``circuit_angles``: :math:`|\langle\mathrm{MPS}_\chi(\text{true})|\mathrm{MPS}_\chi(\theta_i)\rangle|^2`
    / norms, with the true state given by its angles ``prep_theta`` (no dense ``2**n`` state).
    """
    n_qubits = int(n_qubits)
    chi = int(tensor_bond_dim)

    if state_param == "cholesky":
        psi_tgt = np.asarray(psi_true, dtype=np.complex128).reshape(-1)
        out = []
        for i in range(len(samples)):
            psi, _rho = theta_to_psi_and_rho(samples[i], n_qubits=n_qubits, state_param=state_param)
            out.append(float(fidelity_pure(psi_tgt, psi)))
        return np.asarray(out, dtype=np.float64)

    if prep_theta is None:
        raise ValueError("circuit_angles fidelities need prep_theta (the true state's angles)")
    mps_kw = dict(
        n_qubits=n_qubits,
        bond_dim=chi,
        ansatz_layers=ansatz_layers,
        unified_ry_rx=unified_ry_rx,
        cnot_topology=cnot_topology,
    )
    ref_tensors = theta_to_mps_tensors(np.asarray(prep_theta, dtype=np.float64).reshape(-1), **mps_kw)
    out = [
        float(fidelity_mps_tensors(ref_tensors, theta_to_mps_tensors(samples[i], **mps_kw)))
        for i in range(len(samples))
    ]
    return np.asarray(out, dtype=np.float64)


def _angle_ticklabel_tex(x: float, *, abs_tol: float = 1.5e-4) -> str:
    """Mathtext tick for radians: ``0``, ``\\pi/4``, … if ``x`` ≈ ``k\\pi/n`` with small ``n``; else decimal."""
    if not np.isfinite(x):
        return r"$?$"
    pi = math.pi
    if abs(x) < abs_tol:
        return r"$0$"
    best_n = 99
    best_k = 0
    for n in range(1, 17):
        k = int(round(x * n / pi))
        approx = k * pi / n
        if abs(x - approx) <= abs_tol:
            if n < best_n or (n == best_n and abs(k) < abs(best_k)):
                best_n = n
                best_k = k
    if best_n == 99:
        return rf"${x:.4g}$"

    def _tex(k: int, n: int) -> str:
        g = math.gcd(abs(k), n)
        k, n = k // g, n // g
        if k == 0:
            return r"$0$"
        neg = k < 0
        k = abs(k)
        body: str
        if n == 1:
            body = r"\pi" if k == 1 else rf"{k}\pi"
        elif k == 1:
            body = rf"\pi/{n}"
        else:
            body = rf"{k}\pi/{n}"
        return rf"$-{body}$" if neg else rf"${body}$"

    return _tex(best_k, best_n)


def _grid_cell_edges_1d(centers: np.ndarray) -> np.ndarray:
    """Piecewise-uniform bin edges so each cell is centered on ``centers[i]`` (also OK for non-uniform spacing)."""
    c = np.asarray(centers, dtype=np.float64).ravel()
    if c.size == 0:
        return np.array([0.0, 1.0])
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5], dtype=np.float64)
    mid = 0.5 * (c[:-1] + c[1:])
    left = c[0] - (mid[0] - c[0])
    right = c[-1] + (c[-1] - mid[-1])
    return np.concatenate([[left], mid, [right]])


def _sweep_theta_x_from_npz(dat) -> np.ndarray:
    """Vertical sweep angles: ``theta_x`` (legacy checkpoints used ``theta_z``)."""
    if "theta_x" in dat:
        return np.asarray(dat["theta_x"], dtype=np.float64).ravel()
    return np.asarray(dat["theta_z"], dtype=np.float64).ravel()


def plot_prep_angle_sweep_median_infidelity(
    output_dir: str,
    theta_y: np.ndarray,
    theta_x: np.ndarray,
    median_infidelity: np.ndarray,
) -> None:
    """``median_infidelity[iy, ix]`` at ``(theta_y[iy], theta_x[ix])`` — cell-centered heatmap."""
    with plt.rc_context(_FIG_RC):
        fig, ax = plt.subplots(figsize=(3.55, 3.05), constrained_layout=True)
        Z = np.asarray(median_infidelity, dtype=np.float64)
        ty = np.asarray(theta_y, dtype=np.float64).ravel()
        tx = np.asarray(theta_x, dtype=np.float64).ravel()
        ty_edges = _grid_cell_edges_1d(ty)
        tx_edges = _grid_cell_edges_1d(tx)
        Zp = np.clip(Z, 0.0, np.inf)
        zmax = float(np.nanmax(Zp)) if np.isfinite(Zp).any() else 1.0
        if not np.isfinite(zmax) or zmax < 1e-20:
            zmax = 1.0
        norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=0.5 * zmax, vmax=zmax)
        # C[ix, iy] = Z[iy, ix]; x = θ_y, y = θ_x
        im = ax.pcolormesh(
            ty_edges,
            tx_edges,
            Z.T,
            shading="flat",
            cmap=_CMAP_MUTED_DIVERGING,
            norm=norm,
            rasterized=True,
        )
        _tick_fs = 10
        _cbar_fs = 10
        cbar = fig.colorbar(im, ax=ax, shrink=0.74, pad=0.02, aspect=28)
        cbar.set_label(r"$1-F_{\mathrm{median}}$", fontsize=_cbar_fs + 2)
        cbar.locator = ticker.MaxNLocator(nbins=5)
        cbar.update_ticks()
        cbar.ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _pos: f"{int(x)}" if float(x).is_integer() else f"{x:g}")
        )
        cbar.ax.tick_params(which="major", length=2.8, width=0.65, labelsize=_tick_fs)
        ax.set_xlabel(r"$\theta_y$", fontsize=15, labelpad=0)
        ax.set_ylabel(r"$\theta_x$", fontsize=15, labelpad=6)
        # ticks on the sweep grid (at most 6 per axis), labelled as π-fractions
        def _pick_ticks(vals: np.ndarray, max_ticks: int = 6) -> np.ndarray:
            v = np.asarray(vals, dtype=np.float64).ravel()
            if v.size <= max_ticks:
                return v
            stride = max(1, int(round(v.size / float(max_ticks))))
            idx = list(range(0, v.size, stride))
            if len(idx) > max_ticks:
                idx = idx[:max_ticks]
            idx = np.unique(np.asarray(idx, dtype=int))
            if idx.size < 2:
                idx = np.unique(np.linspace(0, v.size - 1, num=max_ticks, dtype=int))
            return v[idx]

        xt = _pick_ticks(ty, max_ticks=6)
        yt = _pick_ticks(tx, max_ticks=6)
        ax.set_xticks(xt)
        ax.set_yticks(yt)
        ax.set_xticklabels([_angle_ticklabel_tex(float(t)) for t in xt], fontsize=_tick_fs)
        ax.set_yticklabels([_angle_ticklabel_tex(float(t)) for t in yt], fontsize=_tick_fs)
        ax.tick_params(direction="out", length=4, width=0.9, labelsize=_tick_fs)
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        ax.set_aspect("equal")
        # white cell boundaries mark the discrete prep grid
        if ty.size <= 24 and tx.size <= 24:
            ax.set_xticks(ty_edges, minor=True)
            ax.set_yticks(tx_edges, minor=True)
            ax.tick_params(which="minor", length=0)
            ax.grid(which="minor", color="white", linewidth=0.55, alpha=0.65)
        path = os.path.join(output_dir, "qst_median_infidelity_landscape.pdf")
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")


def _fidelity_stat_legend_label(name: str, value: float) -> str:
    return rf"$F_{{\mathrm{{{name}}}}}={value:.4f}$"


def _hist_view_quantile(fidelity: np.ndarray) -> dict | None:
    """Histogram view of ``plot_all``: 0.1%–99.9% window, 28 bins, median line."""
    f = np.asarray(fidelity, dtype=np.float64).ravel()
    f = f[np.isfinite(f)]
    if f.size == 0:
        return None
    f_median = float(np.median(f))
    lo, hi = float(np.quantile(f, 0.001)), float(np.quantile(f, 0.999))
    span = max(hi - lo, 1e-12)
    pad = max(0.002, 0.08 * span)
    xlo = max(0.0, lo - pad)
    xhi = min(1.0, hi + pad)
    stat_margin = max(0.002, 0.03 * max(xhi - xlo, 1e-12))
    xlo = max(0.0, min(xlo, f_median - stat_margin))
    n_bins = 28
    return {
        "f": f,
        "hist_vals": f,
        "bin_edges": np.linspace(xlo, xhi, n_bins + 1),
        "xlo": xlo,
        "xhi": xhi,
        "f_median": f_median,
        "f_max": None,
        "n_bins": n_bins,
    }


def _hist_view_tukey(fidelity: np.ndarray, whisker: float = 1.5) -> dict | None:
    """Histogram view of pooled fidelities: samples inside the Tukey fences Q1-k·IQR … Q3+k·IQR, median and max."""
    f = np.asarray(fidelity, dtype=np.float64).ravel()
    f = f[np.isfinite(f)]
    if f.size == 0:
        return None
    q1, q3 = np.quantile(f, [0.25, 0.75])
    iqr = float(q3 - q1)
    core = f[(f >= q1 - whisker * iqr) & (f <= q3 + whisker * iqr)]
    hist_vals = core if core.size else f
    v_lo, v_hi = float(np.min(hist_vals)), float(np.max(hist_vals))
    span = max(v_hi - v_lo, 1e-12)
    pad = _PANEL_X_SIDE_PAD_FRAC * span
    xlo = max(0.0, v_lo - pad)
    xhi = v_hi + pad
    n_bins = int(np.clip(span / 0.00012, 28, 96))
    return {
        "f": f,
        "hist_vals": hist_vals,
        "bin_edges": np.linspace(xlo, xhi, n_bins + 1),
        "xlo": xlo,
        "xhi": xhi,
        "f_median": float(np.median(f)),
        "f_max": float(np.max(f)),
        "n_bins": n_bins,
    }


def _draw_fidelity_histogram_on_ax(ax: plt.Axes, view: dict, *, panel: bool = False, y_log: bool = False) -> None:
    """Density histogram with dashed median (and max) lines; ``panel`` selects the two-panel-figure style."""
    if panel:
        bar_color = mcolors.to_rgba(COMBINED_BAR_COLORS["mean"], alpha=0.6)
        median_color = _MARKER_EDGES["mean"]
        max_color = _MARKER_EDGES["truth"]
        line_lw = 1.5
    else:
        bar_color = "#6baed6"
        median_color = "#08519c"
        max_color = COMBINED_BAR_COLORS["truth"]
        line_lw = 1.0
    ax.hist(
        view["hist_vals"],
        bins=view["bin_edges"],
        density=True,
        color=bar_color,
        edgecolor="white",
        linewidth=0.4,
        alpha=1.0 if panel else 0.92,
        log=y_log,
    )
    ax.axvline(
        view["f_median"],
        color=median_color,
        ls="--",
        lw=line_lw,
        zorder=5,
        label=_fidelity_stat_legend_label("median", view["f_median"]),
    )
    if view["f_max"] is not None:
        ax.axvline(
            view["f_max"],
            color=max_color,
            ls="--",
            lw=line_lw,
            zorder=5,
            label=_fidelity_stat_legend_label("max", view["f_max"]),
        )
    ax.set_xlabel(r"Fidelity $F$", fontsize=_QST_FIDELITY_HIST_LABEL_FONTSIZE)
    ax.set_ylabel("Density", fontsize=_QST_FIDELITY_HIST_LABEL_FONTSIZE)
    ax.set_xlim(view["xlo"], view["xhi"])
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _pos: f"{x:g}"))
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=_QST_FIDELITY_HIST_TICK_LENGTH,
        width=_QST_FIDELITY_HIST_TICK_WIDTH,
        labelsize=_QST_FIDELITY_HIST_TICK_FONTSIZE,
    )
    if panel:
        ax.legend(
            frameon=False,
            loc="upper left",
            fontsize=_QST_FIDELITY_HIST_LEGEND_FONTSIZE,
            borderpad=0.2,
            labelspacing=0.2,
            handlelength=1.4,
        )
    else:
        ax.legend(frameon=False, loc="upper left", fontsize=_QST_FIDELITY_HIST_LEGEND_FONTSIZE)
    if y_log:
        y_lo, y_hi = ax.get_ylim()
        ax.set_ylim(max(y_lo, 1e-2), y_hi * (1.08 if panel else 1.12))
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=5))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(_plain_decimal_log_formatter))
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    else:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _pos: f"{y:g}"))
        if panel:
            ax.margins(y=0.04)
            y0, y1 = ax.get_ylim()
            if y1 > 0:
                ax.set_ylim(0.0, y1 * 1.02)


def _plot_fidelity_histogram(output_dir: str, fidelity: np.ndarray, *, filename: str = "qst_fidelity_hist.pdf") -> None:
    """``qst_fidelity_hist.pdf``: density histogram of the posterior-sample fidelities with the median line."""
    view = _hist_view_quantile(fidelity)
    if view is None:
        return
    with plt.rc_context(_FIG_RC):
        fig, ax = plt.subplots(figsize=(2.75, 2.35), constrained_layout=True)
        _draw_fidelity_histogram_on_ax(ax, view)
        path = os.path.join(output_dir, filename)
        fig.savefig(path)
        plt.close(fig)
    f = view["f"]
    in_window = float(((f >= view["xlo"]) & (f <= view["xhi"])).mean()) * 100.0
    print(
        f"  Saved {path}  (N={f.size}, F_median={view['f_median']:.4f}, "
        f"x=[{view['xlo']:.4g},{view['xhi']:.4g}], bins={view['n_bins']}, in-window={in_window:.1f}%)"
    )


def plot_random_state_fidelity_histogram(
    output_dir: str,
    fidelity: np.ndarray,
    *,
    filename: str = "qst_random_state_fidelity_hist.pdf",
    y_log: bool = True,
    tukey_whisker: float = 1.5,
) -> None:
    """Density histogram of pooled posterior-sample fidelities (random eval or prep-angle sweep grid).

    Samples beyond ``tukey_whisker`` × IQR outside the quartiles are left out of the view; dashed lines
    mark the median and the maximum over all samples.
    """
    view = _hist_view_tukey(fidelity, tukey_whisker)
    if view is None:
        return
    with plt.rc_context(_PANEL_TRAINING_DATA_LINE_RC):
        fig, ax = plt.subplots(figsize=(2.45, 2.25), constrained_layout=True)
        _draw_fidelity_histogram_on_ax(ax, view, panel=True, y_log=y_log)
        _apply_training_data_line_axis_style(ax, panel=True, legend_fontsize=11.0)
        path = os.path.join(output_dir, filename)
        fig.savefig(path)
        plt.close(fig)
    print(
        f"  Saved {path}  (N={view['f'].size}, F_median={view['f_median']:.4f}, "
        f"F_max={view['f_max']:.4f}, x=[{view['xlo']:.4g},{view['xhi']:.4g}], bins={view['n_bins']})"
    )


def plot_all(
    results_dir: str,
    posterior_samples: np.ndarray,
    psi_true: np.ndarray | None,
    *,
    state_param: str,
    tensor_bond_dim: int = 2,
    max_n_qubits_dense_rho_plots: int | None = None,
    prep_theta: np.ndarray | None = None,
    ansatz_layers: int = 1,
    unified_ry_rx: bool = False,
    cnot_topology: str = "linear",
    n_qubits: int | None = None,
) -> dict[str, float]:
    """Fidelity histogram and (for small n) Re/Im ρ heatmaps of the posterior against the true state.

    ``psi_true`` (dense) is only needed for ``cholesky`` and for the ρ heatmaps; for ``circuit_angles``
    the true state is given by ``prep_theta`` and ``n_qubits``. Returns the fidelity statistics.
    """
    if psi_true is not None:
        d = psi_true.size
        n_qubits = int(round(math.log2(d))) if d > 0 else 0
    elif n_qubits is None:
        raise ValueError("plot_all: need n_qubits when psi_true is None")
    else:
        n_qubits = int(n_qubits)
        d = 2**n_qubits

    fids = posterior_sample_fidelities(
        posterior_samples,
        psi_true,
        n_qubits=n_qubits,
        state_param=state_param,
        tensor_bond_dim=tensor_bond_dim,
        prep_theta=prep_theta,
        ansatz_layers=ansatz_layers,
        unified_ry_rx=unified_ry_rx,
        cnot_topology=cnot_topology,
    )
    fidelity_best = float(np.max(fids)) if len(fids) else 0.0
    fidelity_median = float(np.median(fids)) if len(fids) else 0.0
    fidelity_mean_samples = float(np.mean(fids)) if len(fids) else 0.0

    _plot_fidelity_histogram(results_dir, fids, filename="qst_fidelity_hist.pdf")

    dense_limit = (
        _DEFAULT_MAX_N_QUBITS_DENSE_RHO_PLOTS
        if max_n_qubits_dense_rho_plots is None
        else int(max_n_qubits_dense_rho_plots)
    )
    if n_qubits > dense_limit:
        print(f"  Skipping dense ρ figures (n_qubits={n_qubits} > {dense_limit}, d={d}).")
        print(f"  Figures saved under {results_dir}/")
        return {
            "fidelity_mean_samples": fidelity_mean_samples,
            "fidelity_best": fidelity_best,
            "fidelity_median": fidelity_median,
        }

    rho_true = rho_from_psi(psi_true)

    # posterior sample with the lower-median fidelity
    order = np.argsort(fids)
    mid_rank = (len(order) - 1) // 2
    mid_idx = int(order[mid_rank])
    _psi_median_rank, rho_median = theta_to_psi_and_rho(
        posterior_samples[mid_idx],
        n_qubits=n_qubits,
        state_param=state_param,
        ansatz_layers=ansatz_layers,
        unified_ry_rx=unified_ry_rx,
        cnot_topology=cnot_topology,
    )
    fidelity_median_sample = float(fids[mid_idx])

    # Re(ρ), Im(ρ): rows; columns: true | posterior median
    re_t = np.real(rho_true)
    re_m = np.real(rho_median)
    im_t = np.imag(rho_true)
    im_m = np.imag(rho_median)

    def _symm_vlim(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        m = float(max(np.abs(a).max(), np.abs(b).max()))
        if m < 1e-20 or not np.isfinite(m):
            m = 1.0
        return -m, m

    vmin_r, vmax_r = _symm_vlim(re_t, re_m)
    vmin_i, vmax_i = _symm_vlim(im_t, im_m)
    kw_r = dict(
        cmap=_CMAP_MUTED_DIVERGING,
        interpolation="nearest",
        vmin=vmin_r,
        vmax=vmax_r,
        rasterized=True,
    )
    kw_i = dict(
        cmap=_CMAP_MUTED_DIVERGING,
        interpolation="nearest",
        vmin=vmin_i,
        vmax=vmax_i,
        rasterized=True,
    )

    with plt.rc_context(_FIG_RC):
        _tick_fs = 9
        _ket_fs = 8.0
        _cbar_fs = 10

        fig, axes = plt.subplots(2, 2, figsize=_rho_real_imag_figsize(n_qubits), constrained_layout=False)
        fig.subplots_adjust(wspace=0.03, hspace=0.10)
        axes[0, 0].imshow(re_t, **kw_r)
        axes[0, 0].set_title("Ideal", fontsize=_tick_fs)
        im_r1 = axes[0, 1].imshow(re_m, **kw_r)
        axes[0, 1].set_title(rf"$F_{{\mathrm{{median}}}}={fidelity_median_sample:.4f}$", fontsize=_tick_fs)
        axes[1, 0].imshow(im_t, **kw_i)
        axes[1, 0].set_title("")
        im_i1 = axes[1, 1].imshow(im_m, **kw_i)
        axes[1, 1].set_title("")

        for r in range(2):
            for c in range(2):
                ax = axes[r, c]
                ax.set_aspect("equal")
                _set_rho_axes_ket_ticks(ax, n_qubits, label_fontsize=_ket_fs)

        cbr = fig.colorbar(im_r1, ax=axes[0, :], shrink=0.76, pad=0.015, aspect=22)
        cbr.set_label(r"$\mathrm{Re}(\rho)$", fontsize=_cbar_fs)
        cbr.locator = ticker.MaxNLocator(nbins=5)
        cbr.update_ticks()
        cbr.ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _pos: f"{int(x)}" if float(x).is_integer() else f"{x:g}")
        )
        cbr.ax.tick_params(which="major", length=2.8, width=0.65, labelsize=_tick_fs)

        cbi = fig.colorbar(im_i1, ax=axes[1, :], shrink=0.76, pad=0.015, aspect=22)
        cbi.set_label(r"$\mathrm{Im}(\rho)$", fontsize=_cbar_fs)
        cbi.locator = ticker.MaxNLocator(nbins=5)
        cbi.update_ticks()
        cbi.ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _pos: f"{int(x)}" if float(x).is_integer() else f"{x:g}")
        )
        cbi.ax.tick_params(which="major", length=2.8, width=0.65, labelsize=_tick_fs)

        fig.savefig(os.path.join(results_dir, "qst_rho_real_imag.pdf"))
        plt.close(fig)

    print(f"  Figures saved under {results_dir}/")
    return {
        "fidelity_mean_samples": fidelity_mean_samples,
        "fidelity_best": fidelity_best,
        "fidelity_median": fidelity_median,
    }


def _plot_pec_style_param_line(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    param_style: str,
    *,
    label: str | None = None,
    zorder: int,
    markersize_scale: float = 1.0,
) -> mlines.Line2D | None:
    """Line + markers matching ``figure_1x3`` PEC/ZNE curves (pershots256)."""
    face = COMBINED_BAR_COLORS[param_style]
    edge = _MARKER_EDGES[param_style]
    marker = _MARKERS[param_style]
    ms = param_line2d_markersize_pec(param_style) * markersize_scale
    mew = 0.9 * markersize_scale
    ax.plot(
        x,
        y,
        marker="None",
        linestyle="-",
        color=face,
        linewidth=2.2,
        alpha=0.6,
        zorder=zorder,
    )
    ax.plot(
        x,
        y,
        marker=marker,
        linestyle="None",
        markersize=ms,
        markerfacecolor=face,
        markeredgecolor=edge,
        markeredgewidth=mew,
        zorder=zorder,
    )
    if label is None:
        return None
    line_color = mcolors.to_rgba(face, alpha=0.6)
    return mlines.Line2D(
        [],
        [],
        color=line_color,
        linestyle="-",
        linewidth=2.2,
        marker=marker,
        markersize=ms,
        markerfacecolor=face,
        markeredgecolor=edge,
        markeredgewidth=mew,
        label=label,
    )


def _load_sim_budget_entries(
    result_dirs: list[str] | str,
) -> tuple[list[str], list[tuple[int, np.ndarray, int, str]]]:
    import json

    if isinstance(result_dirs, str):
        import glob

        roots = sorted(glob.glob(result_dirs))
    else:
        roots = [os.path.abspath(os.path.expanduser(p)) for p in result_dirs]

    entries: list[tuple[int, np.ndarray, int, str]] = []
    for root in roots:
        cfg_path = os.path.join(root, "config_checkpoint.json")
        npz_path = os.path.join(root, "random_parameter_eval.npz")
        if not (os.path.isfile(cfg_path) and os.path.isfile(npz_path)):
            print(f"  Skipping {root} (missing config or random_parameter_eval.npz)")
            continue
        with open(cfg_path, "r") as f:
            ckpt = json.load(f)
        cfg = ckpt.get("config", ckpt)
        with np.load(npz_path) as zf:
            fids = np.asarray(zf["fidelity"], dtype=np.float64).ravel()
            fids = fids[np.isfinite(fids)]
            n_post = int(zf["num_posterior_samples"])
        if fids.size == 0 or n_post <= 0:
            continue
        entries.append((int(cfg["NPE_N_SIMS"]), fids, n_post, os.path.basename(root)))
    return roots, entries


def _filter_sim_budget_entries(
    entries: list[tuple[int, np.ndarray, int, str]],
    budget_units: tuple[int, ...] = (1, 3, 5, 7, 9),
) -> list[tuple[int, np.ndarray, int, str]]:
    """Keep the runs whose budget is ``u × 10⁵`` for ``u`` in ``budget_units``, in that order."""
    order_map = {u * 100_000: i for i, u in enumerate(budget_units)}
    entries = [e for e in entries if e[0] in order_map]
    return sorted(entries, key=lambda t: order_map[t[0]])


def _draw_sim_budget_line_on_ax(ax: plt.Axes, entries: list[tuple[int, np.ndarray, int, str]]) -> None:
    """Median and maximum fidelity of each run against its simulation budget (two-panel-figure style)."""
    budgets = np.asarray([b for b, _, _, _ in entries], dtype=np.float64)
    medians = np.asarray([float(np.median(f)) for _, f, _, _ in entries], dtype=np.float64)
    maxes = np.asarray([float(np.max(f)) for _, f, _, _ in entries], dtype=np.float64)
    x_plot = budgets / 1.0e5
    handles = [
        _plot_pec_style_param_line(
            ax, x_plot, maxes, "truth", label=r"$F_{\mathrm{max}}$", zorder=1, markersize_scale=0.6,
        ),
        _plot_pec_style_param_line(
            ax, x_plot, medians, "mean", label=r"$F_{\mathrm{median}}$", zorder=2, markersize_scale=0.6,
        ),
    ]
    ax.set_xlabel(r"Simulation budget [$\times 10^{5}$]")
    ax.set_ylabel(r"Fidelity $F$")
    ax.set_xticks(x_plot)
    ax.set_xticklabels([str(int(round(v))) for v in x_plot])
    y_line = np.concatenate([medians, maxes])
    y_lo, y_hi = float(np.min(y_line)), float(np.max(y_line))
    pad = max(0.02, 0.05 * (y_hi - y_lo))
    top_pad = max(0.02, 0.04 * (y_hi - y_lo))
    ax.set_ylim(max(0.0, y_lo - pad), min(1.04, y_hi + top_pad))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{int(v)}" if abs(v - round(v)) < 1e-9 else f"{v:g}")
    )
    x_margin = 0.30
    ax.set_xlim(float(x_plot[0]) - x_margin, float(x_plot[-1]) + x_margin)
    ax.grid(False)
    ax.legend(
        handles,
        [h.get_label() for h in handles],
        frameon=False,
        fontsize=_PANEL_TRAINING_DATA_LINE_RC["legend.fontsize"],
        loc="upper right",
        handlelength=1.4,
        markerscale=1.0,
        borderpad=0.4,
        borderaxespad=0.1,
    )


def plot_sim_budget_and_random_state_fidelity_panels(
    sweep_result_dirs: list[str] | str,
    hist_result_dir: str,
    *,
    output_path: str | None = None,
    budget_units: tuple[int, ...] = (1, 3, 5, 7, 9),
    hist_y_log: bool = True,
    hist_tukey_whisker: float = 1.5,
) -> str:
    """Two-panel figure: median / max fidelity vs simulation budget (left) + random-state fidelity histogram (right).

    ``sweep_result_dirs``: the ``qst_<time>_nsims<N>/`` run directories (or a glob); ``hist_result_dir``:
    the run whose pooled fidelities fill the histogram (the largest budget).
    """
    hist_root = os.path.abspath(os.path.expanduser(hist_result_dir))
    hist_npz = os.path.join(hist_root, "random_parameter_eval.npz")
    if not os.path.isfile(hist_npz):
        raise FileNotFoundError(f"Missing random eval data: {hist_npz}")

    _, entries = _load_sim_budget_entries(sweep_result_dirs)
    entries = _filter_sim_budget_entries(entries, budget_units)
    if not entries:
        raise FileNotFoundError("No valid QST sweep directories with random eval data.")

    with np.load(hist_npz) as zf:
        hist_fids = np.asarray(zf["fidelity"], dtype=np.float64).ravel()
    if output_path is None:
        parent = os.path.dirname(hist_root) or "sbi_qst_results"
        output_path = os.path.join(parent, "qst_random_state_fidelity_sim_budget_hist_panels.pdf")
    output_path = os.path.abspath(os.path.expanduser(output_path))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    panel_w_in = 2.20
    panel_h_in = 2.00
    gap_in = 0.72
    left_in, bottom_in = 0.07, 0.11
    right_pad_in = 0.10
    top_in = 0.03
    fig_w = left_in + 2.0 * panel_w_in + gap_in + right_pad_in
    fig_h = bottom_in + panel_h_in + top_in

    hist_view = _hist_view_tukey(hist_fids, hist_tukey_whisker)
    if hist_view is None:
        raise ValueError(f"No finite fidelity samples in {hist_npz}")

    with plt.rc_context(_PANEL_TRAINING_DATA_LINE_RC):
        fig = plt.figure(figsize=(fig_w, fig_h))
        ax_line = fig.add_axes(
            [left_in / fig_w, bottom_in / fig_h, panel_w_in / fig_w, panel_h_in / fig_h]
        )
        ax_hist = fig.add_axes(
            [
                (left_in + panel_w_in + gap_in) / fig_w,
                bottom_in / fig_h,
                panel_w_in / fig_w,
                panel_h_in / fig_h,
            ]
        )
        _draw_sim_budget_line_on_ax(ax_line, entries)
        _draw_fidelity_histogram_on_ax(ax_hist, hist_view, panel=True, y_log=hist_y_log)
        _apply_training_data_line_axis_style(ax_line, panel=True, legend_fontsize=11.0)
        _apply_training_data_line_axis_style(ax_hist, panel=True, legend_fontsize=11.0)
        _align_panel_xaxis_labels([ax_line, ax_hist], labelpad=3.0)
        fig.savefig(output_path, pad_inches=0.01)
        plt.close(fig)

    print(f"  Saved {output_path}")
    return output_path


def reproduce(results_dir: str) -> dict[str, float]:
    """Redraw the figures of a prep-angle sweep directory (written by ``run_sbi.py``) from its saved files.

    Needs ``config_checkpoint.json`` and ``posterior_samples_reference.csv``; the sweep heatmap and the
    pooled sweep fidelity histogram are drawn when ``sweep_median_infidelity.npz`` /
    ``sweep_fidelity_grid.npz`` exist.
    """
    import json

    root = os.path.abspath(os.path.expanduser(results_dir))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"results_dir not found: {root}")

    with open(os.path.join(root, "config_checkpoint.json"), "r") as f:
        ckpt = json.load(f)
    cfg = ckpt.get("config", ckpt)

    n_qubits = int(cfg["N_QUBITS"])
    state_param = str(cfg["STATE_PARAM"]).lower()
    ty0 = float(cfg.get("PREP_THETA_Y", np.pi / 4.0))
    tx0 = float(cfg.get("PREP_THETA_X", np.pi / 4.0))
    posterior_samples = np.loadtxt(
        os.path.join(root, "posterior_samples_reference.csv"), delimiter=",", dtype=np.float64
    )
    max_dense = int(cfg.get("MAX_N_QUBITS_DENSE_RHO_PLOTS", n_qubits))
    per_qubit = bool(cfg.get("BRICKWALL_PER_QUBIT_PREP", False))
    unified = bool(cfg.get("BRICKWALL_UNIFIED_RY_RX", False))
    layers = max(1, int(cfg.get("BRICKWALL_ANSATZ_LAYERS", 1)))
    topology = str(cfg.get("BRICKWALL_CNOT_TOPOLOGY", "linear")).lower()

    prep_theta = prep_to_theta(
        ty0, tx0, n_qubits=n_qubits, state_param=state_param,
        per_qubit_prep=per_qubit, unified_ry_rx=unified, ansatz_layers=layers,
    )
    if state_param == "cholesky":
        psi_ref = brickwall_rx_ry_state_vector(n_qubits, theta_y=ty0, theta_x=tx0)
    elif n_qubits <= max_dense:
        psi_ref = theta_to_psi_and_rho(
            prep_theta, n_qubits=n_qubits, state_param=state_param,
            ansatz_layers=layers, unified_ry_rx=unified, cnot_topology=topology,
        )[0]
    else:
        psi_ref = None

    print(f"Reproducing QST plots from: {root}")
    print(f"  n_qubits={n_qubits}, PREP_THETA_Y={ty0}, PREP_THETA_X={tx0}")
    print(f"  Posterior samples: {posterior_samples.shape[0]} × {posterior_samples.shape[1]}")

    fstats = plot_all(
        results_dir=root,
        posterior_samples=posterior_samples,
        psi_true=psi_ref,
        state_param=state_param,
        tensor_bond_dim=int(cfg.get("TENSOR_BOND_DIM", 2)),
        max_n_qubits_dense_rho_plots=max_dense,
        prep_theta=prep_theta if state_param == "circuit_angles" else None,
        ansatz_layers=layers,
        unified_ry_rx=unified,
        cnot_topology=topology,
        n_qubits=n_qubits,
    )

    sweep_npz = os.path.join(root, "sweep_median_infidelity.npz")
    if os.path.isfile(sweep_npz):
        dat = np.load(sweep_npz)
        plot_prep_angle_sweep_median_infidelity(
            root,
            theta_y=np.asarray(dat["theta_y"], dtype=np.float64),
            theta_x=_sweep_theta_x_from_npz(dat),
            median_infidelity=np.asarray(dat["median_infidelity"], dtype=np.float64),
        )
    else:
        print(f"  Note: {sweep_npz} not found; skipping sweep heatmap.")

    sweep_fid_npz = os.path.join(root, "sweep_fidelity_grid.npz")
    if os.path.isfile(sweep_fid_npz):
        with np.load(sweep_fid_npz) as zf:
            plot_random_state_fidelity_histogram(root, zf["fidelity"], filename="qst_sweep_fidelity_hist.pdf")
    else:
        print(f"  Note: {sweep_fid_npz} not found; skipping pooled sweep fidelity histogram.")

    return fstats
