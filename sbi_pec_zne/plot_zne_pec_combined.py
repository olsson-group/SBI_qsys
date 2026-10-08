"""Combined 1×3 figure of the ZNE and PEC results.

Columns: ZNE extrapolation panels (one per method) | PEC convergence with the sampling overhead as an
inset | bar chart of the ZNE estimates and the PEC estimate at one sample size.

Usage (from the repository root):
    python -m sbi_pec_zne.plot_zne_pec_combined [results_dir] [--pec-sample-size 2000] [--out figure_1x3.pdf]

results_dir (default sbi_pec_zne_results/pershots256) holds zne/zne_results.json and
pec/pec_results.json; the figure is saved inside it.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from matplotlib.legend_handler import HandlerTuple

from sbi_pec_zne.pec.visualize import (
    draw_pec_overhead_on_ax,
    draw_pec_single_observation_on_ax,
)
from sbi_pec_zne.style import (
    EDGE, IDEAL_STYLE, LABEL, MARKER, NOISY_STYLE, VIZ_RC,
    fill, markersize, ordered_names,
)
from sbi_pec_zne.zne.visualize import draw_zne_extrapolation_on_ax

_VIZ_RC = {**{k: v for k, v in VIZ_RC.items() if k != "axes.titlesize"}, "legend.fontsize": 16}

_METHOD_SHORT = {"linear": "ZNE\n(lin)", "quadratic": "ZNE\n(quad)", "exponential": "ZNE\n(exp)"}


def _pec_xtick_sample_suffix(n: int) -> str:
    """Second line under ``PEC``: sample size in ``k`` units, e.g. 2000 → ``(2k)``."""
    n = int(n)
    if n == 0:
        return "(0)"
    sign = "-" if n < 0 else ""
    n_abs = abs(n)
    if n_abs < 1000:
        return f"({sign}{n_abs})"
    if n_abs % 1000 == 0:
        return f"({sign}{n_abs // 1000}k)"
    return f"({sign}{n_abs / 1000:g}k)"


def _legend_handles(zne_results: dict) -> tuple[list, list]:
    """One legend row: [bar patch | marker] per parameter summary, then the Ideal / Noisy lines."""
    handles: list = []
    labels: list[str] = []
    for pn in ordered_names(list(zne_results)):
        c = fill(pn)
        h_mark = mlines.Line2D(
            [], [], linestyle="None", marker=MARKER.get(pn, "o"), markersize=markersize(pn),
            markerfacecolor=c, markeredgecolor=EDGE.get(pn, "black"), markeredgewidth=0.8,
        )
        h_bar = mpatches.Rectangle(
            (0, 0), 1, 1, facecolor=c, edgecolor="black", linewidth=0.6,
        )
        handles.append((h_bar, h_mark))
        labels.append(LABEL.get(pn, pn.capitalize()))
    handles.append(mlines.Line2D([], [], **IDEAL_STYLE))
    labels.append("Ideal")
    handles.append(mlines.Line2D([], [], **NOISY_STYLE))
    labels.append("Noisy")
    return handles, labels


def _pec_value_at(pec_data: dict, name: str, obs: str, sample_size: int) -> float:
    """Return PEC expectation for `name` (truth/median/mean/…) at `sample_size`."""
    checkpoints = pec_data["checkpoints"]
    idx = checkpoints.index(sample_size)
    return pec_data["results"][name][obs][idx]


def plot_combined_on_ax(
    ax,
    zne_data: dict,
    pec_data: dict,
    *,
    pec_sample_size: int = 2000,
) -> None:
    """Bar chart of the ZNE estimates (one group per extrapolation method) and the PEC estimate."""
    obs = zne_data["observables"][0]
    methods = zne_data["extrapolation_methods"]
    ideal = zne_data["ideal_expectations"][obs]
    noisy = zne_data["noisy_expectations"][obs]
    zne_results = zne_data["results"]

    param_names = [k for k in ("truth", "median", "mean") if k in zne_results]
    n_params = len(param_names)

    zne_vals: dict[str, dict[str, float]] = {}
    for m in methods:
        zne_vals[m] = {}
        for pn in param_names:
            r = zne_results[pn][obs]
            zne_vals[m][pn] = r["extrapolations"][m]["value"]

    pec_vals: dict[str, float] = {}
    for pn in ("truth", "median", "mean"):
        if pn in pec_data["results"]:
            pec_vals[pn] = _pec_value_at(pec_data, pn, obs, pec_sample_size)

    pec_param_names = [k for k in ("truth", "median", "mean") if k in pec_vals]

    n_zne = len(methods)
    group_gap = 1.0
    x_zne = np.arange(n_zne, dtype=float)
    x_pec_center = n_zne - 1 + group_gap
    bar_width = 0.8 / n_params
    all_vals: list[float] = []

    ax.axhline(ideal, label="Ideal", **IDEAL_STYLE)
    ax.axhline(noisy, label="Noisy", **NOISY_STYLE)

    for i, pn in enumerate(param_names):
        offset = (i - (n_params - 1) / 2) * bar_width
        vals = [zne_vals[m][pn] for m in methods]
        all_vals.extend(vals)
        ax.bar(
            x_zne + offset, vals, bar_width,
            label=LABEL[pn], color=fill(pn),
            edgecolor="black", linewidth=0.6,
        )

    n_pec = len(pec_param_names)
    pec_bar_width = 0.8 / n_pec
    for i, pn in enumerate(pec_param_names):
        offset = (i - (n_pec - 1) / 2) * pec_bar_width
        v = pec_vals[pn]
        all_vals.append(v)
        ax.bar(
            x_pec_center + offset, v, pec_bar_width,
            color=fill(pn), edgecolor="black", linewidth=0.6,
            label=f"_pec_{pn}",
        )

    x_all = list(x_zne) + [x_pec_center]
    labels_all = [_METHOD_SHORT[m] for m in methods] + [f"PEC\n{_pec_xtick_sample_suffix(pec_sample_size)}"]
    ax.set_xticks(x_all)
    ax.set_xticklabels(labels_all)

    ref_vals = [ideal, noisy]
    y_min = min(all_vals + ref_vals)
    y_max = max(all_vals + ref_vals)
    y_span = y_max - y_min or 0.1
    ax.set_ylim(y_min - 0.15 * y_span, y_max + 0.15 * y_span)
    ax.yaxis.set_major_locator(mtick.MaxNLocator(5))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))

    ax.set_ylabel("Expectation")
    ax.grid(False)

def plot_figure_1x3(
    zne_data: dict,
    pec_data: dict,
    out_path: str,
    *,
    pec_sample_size: int = 2000,
) -> None:
    """
    Compact 1×3 figure (vector PDF) that reuses all 6 panels:

    - Column 1: one narrow ZNE extrapolation panel per method, packed into the width of one panel.
    - Column 2: PEC convergence with the sampling overhead as an inset.
    - Column 3: bar chart of the ZNE estimates and the PEC estimate.
    """
    methods = zne_data["extrapolation_methods"]
    if len(methods) < 3:
        raise ValueError(
            f"plot_figure_1x3 needs 3 ZNE methods, got {len(methods)}: {methods}"
        )

    obs0 = zne_data["observables"][0]
    gains = zne_data.get("gains", [1.0, 1.2, 1.6])
    zne_results = zne_data["results"]
    ideal_z = zne_data.get("ideal_expectations", {})
    noisy_z = zne_data.get("noisy_expectations", {})

    pec_results = pec_data["results"]
    pec_checkpoints = pec_data["checkpoints"]
    pec_costs = pec_data.get("pec_costs") or {}
    ideal_p = pec_data.get("ideal_expectations")
    noisy_p = pec_data.get("noisy_expectations")
    pec_obs_list = list(next(iter(pec_results.values())).keys())
    pec_param_names = list(pec_results.keys())
    first_obs = pec_obs_list[0]

    ckpt_abs_max = max(abs(float(c)) for c in pec_checkpoints) if pec_checkpoints else 0.0
    x_exp = int(np.floor(np.log10(ckpt_abs_max))) if ckpt_abs_max >= 1000 else 0
    x_scale = 10 ** x_exp
    x_label = "Sample size" if x_exp == 0 else rf"Sample size [$\times 10^{x_exp}$]"
    x_idx = list(range(len(pec_checkpoints)))

    col_w_in = 350.8 / 72
    row_h_in = 263.9 / 72

    with plt.rc_context(_VIZ_RC):
        fig = plt.figure(figsize=(3 * col_w_in, 1.15 * row_h_in))
        gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.34)

        # Column 1: ZNE panels sharing the y axis, with one shared x title below
        zne_gs = gs[0, 0].subgridspec(1, 3, wspace=0.02)
        zne_axes = []
        ax0 = None
        for i, m in enumerate(methods[:3]):
            if i == 0:
                ax = fig.add_subplot(zne_gs[0, i])
                ax0 = ax
            else:
                ax = fig.add_subplot(zne_gs[0, i], sharey=ax0, sharex=ax0)
            zne_axes.append(ax)
            draw_zne_extrapolation_on_ax(
                ax, zne_results, gains, obs0, ideal_z, noisy_z, m,
            )
            # only the left panel keeps the y axis
            if i > 0:
                ax.set_ylabel("")
                ax.tick_params(axis="y", which="both", left=False, labelleft=False)
        zne_axes[0].set_ylabel("Expectation")
        # per-axes x-labels removed; the shared "Noise gain G" title is added below
        for ax in zne_axes:
            ax.set_xlabel("")
        # ticks at x=0 and every gain, rotated to fit
        xt_all = [0.0] + [float(g) for g in gains]
        lbl = [f"{v:g}" for v in xt_all]
        for i_ax, ax in enumerate(zne_axes):
            ax.set_xticks(xt_all)
            ax.set_xticklabels(
                lbl,
                rotation=45,
                ha="right",
                rotation_mode="anchor",
            )
            ax.tick_params(axis="x", which="major", pad=1.5, labelsize=12)
            # keep tick labels inside their own axes
            for tl in ax.get_xticklabels():
                tl.set_clip_on(True)
        # Column 2: PEC convergence with the overhead bars as an inset
        ax_pec = fig.add_subplot(gs[0, 1])
        draw_pec_single_observation_on_ax(
            ax_pec, first_obs, pec_results, pec_checkpoints, pec_param_names,
            ideal_p, noisy_p,
            x_idx=x_idx, x_scale=x_scale, x_label=x_label,
            show_legend=False,
        )
        # inset [x0, y0, w, h] in axes fractions, in the empty bottom-right region of the PEC panel
        ax_oh = ax_pec.inset_axes([0.56, 0.16, 0.40, 0.26])
        draw_pec_overhead_on_ax(ax_oh, pec_param_names, pec_costs)
        ax_oh.set_ylabel("Overhead", fontsize=10, labelpad=1.0)
        ax_oh.yaxis.set_label_coords(-0.22, 0.5)
        ax_oh.tick_params(axis="both", which="major", length=2.8, width=0.7, labelsize=10)
        for spine in ax_oh.spines.values():
            spine.set_linewidth(0.7)

        # Column 3: bar chart
        ax_bar = fig.add_subplot(gs[0, 2])
        plot_combined_on_ax(
            ax_bar, zne_data, pec_data,
            pec_sample_size=pec_sample_size,
        )

        # align the y-label positions across the columns
        ax_pec.yaxis.set_label_coords(-0.20, 0.5)
        ax_bar.yaxis.set_label_coords(-0.20, 0.5)
        zne_axes[0].yaxis.set_label_coords(-0.50, 0.5)

        _tick_x_56 = _VIZ_RC["axes.labelsize"]
        _tick_y_56 = _VIZ_RC["ytick.labelsize"]
        for _ax in (ax_bar,):
            _ax.tick_params(axis="x", which="major", labelsize=_tick_x_56)
            _ax.tick_params(axis="y", which="major", labelsize=_tick_y_56)

        # place the shared x title at the rendered baseline of the PEC x-label
        fig.subplots_adjust(left=0.13, right=0.99, top=0.90, bottom=0.33, wspace=0.34)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        pec_xlab = ax_pec.xaxis.get_label()
        pec_xlab_bb = pec_xlab.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
        xlab_y = pec_xlab_bb.y0

        # centred under the middle ZNE panel
        bb_mid = zne_axes[1].get_position()
        x_center = 0.5 * (bb_mid.x0 + bb_mid.x1)
        fig.text(
            x_center,
            xlab_y,
            "Noise gain $G$",
            ha="center",
            va="baseline",
            fontsize=_VIZ_RC["axes.labelsize"],
        )

        # one-row legend below the x title
        leg_y = max(0.01, xlab_y - 0.125)
        h_leg, lb_leg = _legend_handles(zne_results)
        ncol = len(h_leg)
        leg_fs = int(_VIZ_RC["legend.fontsize"])
        fig.legend(
            h_leg, lb_leg,
            loc="lower center", bbox_to_anchor=(0.5, leg_y),
            ncol=ncol, frameon=False, columnspacing=1.05, handletextpad=0.5,
            handler_map={tuple: HandlerTuple(ndivide=None)},
            fontsize=leg_fs,
            borderaxespad=0.0,
        )
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    abs_path = os.path.abspath(out_path)
    link = f"\033]8;;file://{abs_path}\033\\{abs_path}\033]8;;\033\\"
    print(f"[figure_1x3] Saved → {link}")


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="Combined ZNE + PEC figure")
    parser.add_argument(
        "results_dir", nargs="?",
        default=os.path.join(repo_root, "sbi_pec_zne_results", "pershots256"),
        help="Results directory containing zne/ and pec/",
    )
    parser.add_argument("--pec-sample-size", type=int, default=2000,
                        help="PEC sample size shown in the bar chart (must be one of the checkpoints)")
    parser.add_argument("--out", default="figure_1x3.pdf",
                        help="Output filename (saved inside results_dir)")
    args = parser.parse_args()

    zne_json = os.path.join(args.results_dir, "zne", "zne_results.json")
    pec_json = os.path.join(args.results_dir, "pec", "pec_results.json")
    for p in (zne_json, pec_json):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing: {p}")
    with open(zne_json) as f:
        zne_data = json.load(f)
    with open(pec_json) as f:
        pec_data = json.load(f)

    if args.pec_sample_size not in pec_data["checkpoints"]:
        raise ValueError(
            f"--pec-sample-size {args.pec_sample_size} not in checkpoints {pec_data['checkpoints']}"
        )
    plot_figure_1x3(
        zne_data, pec_data, os.path.join(args.results_dir, args.out),
        pec_sample_size=args.pec_sample_size,
    )


if __name__ == "__main__":
    main()
