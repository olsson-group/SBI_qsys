"""Figures for the lattice SBI run: per-atom posteriors, regression scatter, pooled RMSE."""

from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

ACCENT = "#53a6cb"
POSTERIOR_LINE = "#4878CF"
TRUTH = "#D65F5F"
TICK_FRAC = 4.0 / 7.0  # per-atom axis ticks sit at ±TICK_FRAC of the half-range

# Discrete confidence shading: <20% white, darker blue for higher confidence.
_CONF_BOUNDS = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
_CONF_COLORS = ["#ffffff", "#dcebf2", "#c1ddeb", "#a4cfe4", "#83bfdb"]
_CMAP_CONF = ListedColormap(_CONF_COLORS, name="confidence_5bin")
_NORM_CONF = BoundaryNorm(_CONF_BOUNDS, _CMAP_CONF.N, clip=True)


def _mm_to_in(mm: float) -> float:
    return mm / 25.4


def _rc(**extra) -> dict:
    """Shared Nature-style rcParams; ``extra`` overrides."""
    base = {
        "text.color": "#000000", "axes.edgecolor": "#000000", "axes.labelcolor": "#000000",
        "xtick.color": "#000000", "ytick.color": "#000000",
        "font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
        "font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    base.update(extra)
    return base


def _int_label(v: float, _pos: object = None) -> str:
    return "0" if abs(float(v)) < 1e-12 else str(int(round(float(v))))


# ---------------------------------------------------------------------------
# Regression: scatter and pooled RMSE
# ---------------------------------------------------------------------------

def plot_regression_scatter(
    results_dir: str,
    theta_true: np.ndarray,
    theta_samples: np.ndarray,
    d_max: float,
    fname: str = "regression_scatter.pdf",
) -> tuple[str, float]:
    """Posterior mean ± std vs truth over all targets/params (nm). Returns (path, R²).

    theta_true: (n_targets, n_params) μm; theta_samples: (n_targets, n_samples, n_params) μm.
    """
    hat = theta_samples.mean(axis=1)
    std = theta_samples.std(axis=1)
    abs_err = np.abs(theta_samples - theta_true[:, None, :]).astype(np.float64)
    mae, sd = float(abs_err.mean()), float(abs_err.std())

    true_flat, hat_flat = theta_true.ravel(), hat.ravel()
    ss_tot = float(np.sum((true_flat - true_flat.mean()) ** 2))
    r2 = 1.0 - float(np.sum((true_flat - hat_flat) ** 2)) / ss_tot if ss_tot > 0 else float("nan")

    lim = float(d_max) * 1e3
    width = _mm_to_in(89)
    rc = _rc(**{"mathtext.fontset": "dejavusans", "axes.labelsize": 15,
                "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 10})
    with mpl.rc_context(rc):
        fig, ax = plt.subplots(figsize=(width, width * 0.62))
        fig.patch.set_facecolor("white")
        ax.errorbar(
            true_flat * 1e3, hat_flat * 1e3, yerr=std.ravel() * 1e3,
            fmt="o", color=ACCENT,
            markerfacecolor=mpl.colors.to_rgba(ACCENT, 0.35),
            markeredgecolor=mpl.colors.to_rgba("#2C3E50", 0.55),
            markeredgewidth=0.6, markersize=2.5, linewidth=0,
            elinewidth=0.6, ecolor=ACCENT, alpha=0.5, capsize=1.5, capthick=0.6, zorder=1,
        )
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("True displacement [nm]", fontsize=15)
        ax.set_ylabel("Posterior [nm]", fontsize=15)
        ax.text(
            0.04, 0.93, f"MAE±s.d.: {mae * 1e3:.1f}±{sd * 1e3:.1f} nm\nR²={r2:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=10, linespacing=1.25,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, boxstyle="round,pad=0.25"),
        )
        ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.6, linewidth=0.9, zorder=10)
        ax.tick_params(axis="both", which="both", direction="out", labelsize=12,
                       length=4, width=0.9, top=False, right=False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
        ax.set_xticks([-lim, 0.0, lim])
        ax.set_yticks([-lim, 0.0, lim])
        path = os.path.join(results_dir, fname)
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return path, r2


def plot_regression_pooled_rmse(
    results_dir: str,
    theta_true: np.ndarray,
    theta_samples: np.ndarray,
    d_max: float,
    *,
    fname: str = "rydberg_lattice_regression_pooled_rmse.pdf",
    prior_seed: int = 90210,
) -> str:
    """Histogram of ||θ - θ_true||₂ [nm], posterior vs prior, pooled over regression targets."""
    n_t, n_s, n_p = theta_samples.shape
    rmse_post = np.linalg.norm(theta_samples - theta_true[:, None, :], axis=-1).ravel() * 1000.0
    rng = np.random.default_rng(prior_seed)
    rmse_prior = np.concatenate([
        np.linalg.norm(rng.uniform(-d_max, d_max, size=(n_s, n_p)) - theta_true[i], axis=-1) * 1000.0
        for i in range(n_t)
    ])

    rc = _rc(**{"mathtext.fontset": "stix", "axes.labelsize": 14,
                "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 15})
    with mpl.rc_context(rc):
        fig, ax = plt.subplots(figsize=(_mm_to_in(89), _mm_to_in(89) * 0.62))
        bins = min(40, max(10, len(rmse_post) // 4))
        edges = np.histogram_bin_edges(np.concatenate([rmse_post, rmse_prior]), bins=bins)

        for values, face, line, z, alpha in [
            (rmse_prior, "#C0C0C0", "#1F2D3A", 1, 0.8),
            (rmse_post, ACCENT, "#2C3E50", 2, 0.8),
        ]:
            y, _ = np.histogram(values, bins=edges, density=True)
            bars = ax.bar(edges[:-1], y, width=np.diff(edges), align="edge", color=face, alpha=alpha,
                          edgecolor="none", linewidth=0.0, antialiased=False, zorder=z, rasterized=True)
            for p in bars.patches:
                p.set_edgecolor("none")
                p.set_linewidth(0.0)
                p.set_antialiased(False)
                p.set_snap(True)
            ax.hist(values, bins=edges, histtype="step", density=True, color=line, linewidth=0.7)

        ax.set_xlabel(r"$\|\boldsymbol{\theta} - \boldsymbol{\theta}_{\mathrm{true}}\|_2$ [nm]", fontsize=13)
        ax.set_ylabel("Density")
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))
        ax.yaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda v, _: "0" if np.isclose(v, 0.0) else f"{v:g}"))
        x_ticks = np.array([0.0, 500.0, 1000.0, 1500.0])
        ax.set_xlim(0.0, max(float(edges[-1]), 1580.0))
        ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(x_ticks))
        ax.xaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda v, _: "0" if np.isclose(v, 0.0) else f"{v:g}"))
        ax.legend(
            handles=[
                Patch(facecolor=mpl.colors.to_rgba("#C0C0C0", 0.5), edgecolor="#1F2D3A", linewidth=0.7, label="Prior"),
                Patch(facecolor=mpl.colors.to_rgba(ACCENT, 0.9), edgecolor="#2C3E50", linewidth=0.7, label="Posterior"),
            ],
            frameon=False, fontsize=11,
        )
        ax.grid(False)

        # Narrow the axes horizontally (centred) without changing figure height.
        shrink = 1.0 - 85.0 / 89.0
        pos = ax.get_position()
        new_w = pos.width * (1.0 - shrink)
        ax.set_position([pos.x0 + 0.5 * (pos.width - new_w), pos.y0, new_w, pos.height])

        out = os.path.join(results_dir, fname)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Per-atom posterior grid
# ---------------------------------------------------------------------------

def _gaussian_blur_2d(H: np.ndarray, sigma_bins: float) -> np.ndarray:
    r = int(max(1, round(3 * sigma_bins)))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma_bins) ** 2)
    k /= k.sum()
    H = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, H)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, H)


def _mass_levels(mass2d: np.ndarray, masses=(0.2, 0.4, 0.6, 0.8)) -> list[float]:
    """Density thresholds whose superlevel sets hold the given probability masses."""
    h = np.sort(mass2d.ravel())[::-1]
    cdf = np.cumsum(h)
    return [float(h[min(int(np.searchsorted(cdf, m)), h.size - 1)]) for m in masses]


def _confidence_grid(mass2d: np.ndarray) -> np.ndarray:
    """Per-bin confidence = 1 - (mass of all bins at least as dense); 0 where empty."""
    flat = mass2d.ravel()
    order = np.argsort(flat)[::-1]
    cum = np.empty_like(flat)
    cum[order] = np.cumsum(flat[order])
    return np.where(flat > 0, 1.0 - cum, 0.0).reshape(mass2d.shape)


def _square_limits(pts: np.ndarray, truth: np.ndarray, pad_frac: float = 0.02):
    """Smallest square containing the 1–99 percentile box of pts (plus padding) and the truth."""
    (lx, ly), (hx, hy) = np.percentile(pts, 1.0, axis=0), np.percentile(pts, 99.0, axis=0)
    lx, hx = min(lx, truth[0]), max(hx, truth[0])
    ly, hy = min(ly, truth[1]), max(hy, truth[1])
    sx, sy = max(hx - lx, 1e-5), max(hy - ly, 1e-5)
    lx, hx, ly, hy = lx - pad_frac * sx, hx + pad_frac * sx, ly - pad_frac * sy, hy + pad_frac * sy
    cx, cy = 0.5 * (lx + hx), 0.5 * (ly + hy)
    half = max(0.5 * (hx - lx), 0.5 * (hy - ly), 1e-9, abs(truth[0] - cx), abs(truth[1] - cy)) * 1.005
    return cx - half, cx + half, cy - half, cy + half


def plot_atom_positions_posterior(
    results_dir: str,
    theta_true: np.ndarray,
    posterior_samples: np.ndarray,
    nx: int,
    ny: int,
    *,
    subset: int | None = None,
    fname: str = "rydberg_atom_positions_posterior.pdf",
) -> str:
    """One panel per atom: 2D posterior of (Δx, Δy) in nm, shaded by credibility mass.

    Markers: mean (green circle), median (orange diamond), truth (red star).
    ``subset=m`` draws only the bottom-right m×m patch with per-panel zoomed limits;
    otherwise the full lattice on shared axes.
    """
    n = nx * ny
    truth = np.asarray(theta_true, dtype=np.float64).reshape(n, 2) * 1000.0
    S = np.asarray(posterior_samples, dtype=np.float64)
    n_s = S.shape[0]
    disp = S.reshape(n_s, n, 2) * 1000.0

    zoom = subset is not None
    if zoom:
        if not 1 <= subset <= min(nx, ny):
            raise ValueError(f"subset {subset} must fit in lattice {nx}×{ny}")
        gs_nx = gs_ny = subset
        iy0, ix0 = ny - subset, nx - subset
        panels = [(r, c, (iy0 + r) * nx + ix0 + c) for r in range(subset) for c in range(subset)]
        bins = 64 if n_s >= 2000 else 48
    else:
        gs_nx, gs_ny = nx, ny
        panels = [(iy, ix, iy * nx + ix) for iy in range(ny) for ix in range(nx)]
        bins = 52 if n_s >= 2000 else 40
        pad = float(np.abs(disp).max()) * 1.05 + 1e-6
        shared_edges = np.linspace(-pad, pad, bins + 1)

    fs_tick, fs_label = 14, 20
    fs_bar = 17 if zoom else 14
    with mpl.rc_context(_rc(**{"mathtext.fontset": "dejavusans"})):
        fig, axes = plt.subplots(
            gs_ny, gs_nx,
            figsize=(_mm_to_in(183.0), _mm_to_in(183.0 * (0.76 if zoom else 1.0))),
            sharex=not zoom, sharey=not zoom,
            gridspec_kw={"hspace": 0.26, "wspace": 0.50} if zoom else {"hspace": 0.08, "wspace": 0.08},
        )
        fig.patch.set_facecolor("white")
        axes = np.array(axes).reshape(gs_ny, gs_nx)

        for irow, icol, k in panels:
            ax = axes[irow, icol]
            ax.set_facecolor("white")
            pts = disp[:, k, :]
            if zoom:
                x0, x1, y0, y1 = _square_limits(pts, truth[k])
                edges_x, edges_y = np.linspace(x0, x1, bins + 1), np.linspace(y0, y1, bins + 1)
            else:
                edges_x = edges_y = shared_edges

            H, xe, ye = np.histogram2d(pts[:, 0], pts[:, 1], bins=(edges_x, edges_y))
            H = _gaussian_blur_2d(H, sigma_bins=2.0)
            if H.sum() > 0:
                mass = H / H.sum()
                levels = sorted({lv for lv in _mass_levels(mass) if lv > 0})
                ax.imshow(
                    _confidence_grid(mass).T, origin="lower",
                    extent=(xe[0], xe[-1], ye[0], ye[-1]),
                    cmap=_CMAP_CONF, norm=_NORM_CONF, interpolation="bilinear",
                    alpha=0.95, zorder=1, aspect="equal",
                )
                if len(levels) >= 2:
                    X, Y = np.meshgrid(0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:]), indexing="xy")
                    ax.contour(X, Y, mass.T, levels=levels, colors=POSTERIOR_LINE,
                               linewidths=0.85 if zoom else 0.8, alpha=0.95, zorder=2)

            mean, med = pts.mean(axis=0), np.median(pts, axis=0)
            ax.scatter(*mean, marker="o", s=72 if zoom else 36, facecolors="#74c476",
                       edgecolors="black", linewidths=0.6, zorder=6)
            ax.scatter(*med, marker="D", s=52 if zoom else 28, color="#F5A623",
                       edgecolors="black", linewidths=0.6, zorder=6)
            ax.scatter(*truth[k], marker="*", s=130 if zoom else 60, color=TRUTH,
                       edgecolors="black", linewidths=0.4, zorder=7)

            if zoom:
                cx, cy = 0.5 * (edges_x[0] + edges_x[-1]), 0.5 * (edges_y[0] + edges_y[-1])
                tx, ty = TICK_FRAC * 0.5 * (edges_x[-1] - edges_x[0]), TICK_FRAC * 0.5 * (edges_y[-1] - edges_y[0])
                ax.set_xlim(edges_x[0], edges_x[-1])
                ax.set_ylim(edges_y[0], edges_y[-1])
                ax.set_aspect("equal", adjustable="datalim")
                ax.margins(x=0.0, y=0.0)
                ax.xaxis.set_major_locator(mpl.ticker.FixedLocator([cx - tx, cx + tx]))
                ax.yaxis.set_major_locator(mpl.ticker.FixedLocator([cy - ty, cy + ty]))
                ax.xaxis.set_ticks_position("bottom")
                ax.tick_params(axis="x", which="major", direction="out", labelsize=fs_tick, length=4,
                               width=0.9, pad=3.2, labelbottom=True, labeltop=False, top=False)
                ax.tick_params(axis="y", which="major", direction="out", labelsize=fs_tick, length=4,
                               width=0.9, pad=4.5, labelleft=True, labelright=False, right=False)
            else:
                ax.set_xlim(-pad, pad)
                ax.set_ylim(-pad, pad)
                ax.set_aspect("equal")
                ax.set_xticks([-TICK_FRAC * pad, TICK_FRAC * pad])
                ax.set_yticks([-TICK_FRAC * pad, TICK_FRAC * pad])
                ax.tick_params(axis="both", which="major", direction="out", labelsize=fs_tick, length=2.5,
                               width=0.9, labelbottom=(irow == gs_ny - 1), labelleft=(icol == 0),
                               top=False, right=False)
            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(_int_label))
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(_int_label))
            for spine in ax.spines.values():
                spine.set_linewidth(0.7)

        if zoom:
            fig.subplots_adjust(left=0.13, right=0.74, bottom=0.15, top=0.88, wspace=0.50, hspace=0.28)
            fig.legend(
                handles=[
                    mpl.lines.Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#74c476",
                                     markeredgecolor="black", markeredgewidth=0.6, markersize=9, label="Mean"),
                    mpl.lines.Line2D([0], [0], marker="D", linestyle="none", markerfacecolor="#F5A623",
                                     markeredgecolor="black", markeredgewidth=0.6, markersize=8, label="Median"),
                    mpl.lines.Line2D([0], [0], marker="*", linestyle="none", markerfacecolor=TRUTH,
                                     markeredgecolor="black", markeredgewidth=0.4, markersize=13, label="Truth"),
                ],
                loc="upper center", bbox_to_anchor=(0.435, 1.0), ncol=3, frameon=False,
                fontsize=fs_bar, handletextpad=0.3, columnspacing=0.7,
            )

        fig.supxlabel(r"$\Delta x$ [nm]", fontsize=fs_label, y=0.044 if zoom else 0.03)
        fig.supylabel(r"$\Delta y$ [nm]", fontsize=fs_label, x=0.028 if zoom else 0.012)

        # Confidence colour bar shared by all panels.
        cax = fig.add_axes([0.795, 0.20, 0.012, 0.50] if zoom else [0.935, 0.26, 0.016, 0.48])
        for lo, hi, c in zip(_CONF_BOUNDS[:-1], _CONF_BOUNDS[1:], _CONF_COLORS):
            cax.add_patch(mpl.patches.Rectangle((0.0, lo), 1.0, hi - lo, facecolor=c, edgecolor="none"))
        cax.set_xlim(0, 1)
        cax.set_ylim(0, 1)
        cax.set_xticks([])
        bounds = [0.2, 0.4, 0.6, 0.8, 1.0]
        cax.set_yticks(bounds)
        cax.set_yticklabels([f"{b:g}" for b in bounds], fontsize=fs_tick)
        cax.yaxis.tick_right()
        cax.yaxis.set_label_position("right")
        cax.tick_params(axis="y", which="major", direction="out", length=4, width=0.9, pad=2.5, labelsize=fs_tick)
        cax.set_ylabel("Confidence", fontsize=fs_bar, labelpad=6)
        for spine in cax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.7)

        if zoom:  # keep tick numerals from being clipped by neighbouring axes
            for lb in cax.get_yticklabels():
                lb.set_clip_on(False)
                lb.set_zorder(100000)
            cax.yaxis.label.set_clip_on(False)
            for ax in axes.ravel():
                for lb in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
                    lb.set_visible(True)
                    lb.set_clip_on(False)
                    lb.set_zorder(100000)

        out = os.path.join(results_dir, fname)
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return out


def plot_all_lattice(
    results_dir: str,
    *,
    theta_true: np.ndarray,
    posterior_samples: np.ndarray,
    nx: int,
    ny: int,
) -> list[str]:
    """Full-lattice and 3×3 posterior figures; writes figures_index.txt."""
    paths = [plot_atom_positions_posterior(results_dir, theta_true, posterior_samples, nx, ny)]
    m = min(3, nx, ny)
    if m < nx or m < ny:
        paths.append(plot_atom_positions_posterior(
            results_dir, theta_true, posterior_samples, nx, ny,
            subset=m, fname="rydberg_atom_positions_posterior_subset3x3_axes_tl.pdf",
        ))
    with open(os.path.join(results_dir, "figures_index.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(paths))
    return paths
