"""
ZNE visualization: ⟨O⟩ vs noise gain G, extrapolation to G=0.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

from sbi_pec_zne.style import (
    EDGE, IDEAL_STYLE, LABEL, MARKER, NOISY_STYLE, VIZ_RC,
    fill, markersize, ordered_names, scatter_area,
)

_METHOD_SHORT = {"linear": "Lin", "quadratic": "Quad", "exponential": "Exp"}


def _fit_curve(fit_info: dict, g: np.ndarray):
    """Fitted function evaluated on ``g`` from a ZNE fit_info, or None if it has no parameters."""
    p = fit_info.get("params")
    method = fit_info.get("method")
    if not p:
        return None
    if method == "exponential":
        return p["a"] * np.exp(p["b"] * g) + p["c"]
    if method == "linear":
        return p["slope"] * g + p["intercept"]
    if method == "quadratic":
        return p["a"] * g ** 2 + p["b"] * g + p["c"]
    return None


def draw_zne_extrapolation_on_ax(
    ax,
    results: dict,
    gains: list,
    obs0: str,
    ideal_expectations: dict,
    noisy_expectations: dict,
    current_method: str,
) -> None:
    """Draw one ZNE extrapolation panel (one extrapolation method) onto ``ax``."""
    ideal_val = ideal_expectations.get(obs0, np.nan)
    if not np.isnan(ideal_val):
        ax.axhline(ideal_val, label="Ideal", **IDEAL_STYLE)
    noisy_val = noisy_expectations.get(obs0, np.nan)
    if not np.isnan(noisy_val):
        ax.axhline(noisy_val, label="Noisy", **NOISY_STYLE)
    ax.axvline(0, color="#333", linestyle="--", alpha=0.5)
    z_base = 3
    for i_name, name in enumerate(ordered_names(list(results))):
        ev_data = results[name]
        r = ev_data.get(obs0, {})
        if not r:
            continue
        ev_by_gain = r.get("ev_by_gain", {})
        extrapolations = r.get("extrapolations", {})
        if not extrapolations:
            extrapolations = {r.get("fit_info", {}).get("method", "exponential"): {
                "value": r.get("extrapolated", np.nan), "fit_info": r.get("fit_info", {}),
            }}
        face = fill(name)
        marker = MARKER.get(name, "o")
        edge = EDGE.get(name, "#1E293B")
        ev_vals = [ev_by_gain.get(g, ev_by_gain.get(str(g), np.nan)) for g in gains]
        z = z_base + i_name
        s_pt = scatter_area(name)
        ax.scatter(gains, ev_vals, c=face, s=s_pt, marker=marker, edgecolors=edge, linewidths=1, zorder=z)
        if current_method in extrapolations:
            extrap = extrapolations[current_method].get("value", np.nan)
            if not np.isnan(extrap):
                ax.scatter([0], [extrap], c=face, s=s_pt * (100.0 / 60.0), marker="*", edgecolors=edge, linewidths=1, zorder=z + 0.2)
            g_curve = np.linspace(0, max(gains) * 1.05, 80)
            y_curve = _fit_curve(extrapolations[current_method].get("fit_info", {}), g_curve)
            if y_curve is not None:
                ax.plot(g_curve, y_curve, color=face, alpha=0.6, linestyle="-", linewidth=2, zorder=z - 0.1)
        ax.plot([], [], marker=marker, linestyle="", label=LABEL.get(name, name.capitalize()),
                markerfacecolor=face, markeredgecolor=edge, markeredgewidth=1,
                markersize=markersize(name))
    xticks = [0] + list(gains)
    ax.set_xticks(xticks[:7])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel(r"Noise gain $G$")
    ax.yaxis.set_major_locator(mtick.MaxNLocator(4))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.grid(False)


def get_zne_extrapolation_legend_handles(ax, results: dict) -> tuple[list, list]:
    """Handles/labels for ZNE extrapolation (params + Ideal/Noisy), same order as the two-column legend."""
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    _fixed_right = ["Ideal", "Noisy"]
    _left_order = [LABEL.get(p, p.capitalize()) for p in ordered_names(list(results))]
    left_items = [(by_label[lb], lb) for lb in _left_order if lb in by_label]
    right_items = [(by_label[lb], lb) for lb in _fixed_right if lb in by_label]
    n_rows = max(len(left_items), len(right_items))
    _dummy = mlines.Line2D([], [], color="none", label="")
    while len(right_items) < n_rows:
        right_items.append((_dummy, ""))
    h_leg = [h for h, _ in left_items] + [h for h, _ in right_items]
    lb_leg = [lb for _, lb in left_items] + [lb for _, lb in right_items]
    return h_leg, lb_leg


def add_zne_extrapolation_legend_to_first_axis(ax, results: dict) -> None:
    """Two-column legend (params | Ideal/Noisy) on the first ZNE extrapolation axis."""
    h_leg, lb_leg = get_zne_extrapolation_legend_handles(ax, results)
    ax.legend(h_leg, lb_leg, loc="lower left", ncol=2, frameon=False, columnspacing=0.8)


def plot_zne_extrapolation(
    results: dict,
    gains: list,
    output_dir: str,
    observables: Optional[list] = None,
    ideal_expectations: Optional[dict] = None,
    noisy_expectations: Optional[dict] = None,
    extrapolation_methods: Optional[list] = None,
    out_filename: str = "zne_extrapolation.pdf",
) -> str:
    """
    Plot ⟨O⟩ vs gain G with extrapolation to G=0, plus bar chart comparing methods.
    """
    os.makedirs(output_dir, exist_ok=True)
    observables = observables or list(next(iter(results.values())).keys())
    ideal_expectations = ideal_expectations or {}
    noisy_expectations = noisy_expectations or {}
    methods = extrapolation_methods or ["exponential"]

    with plt.rc_context(VIZ_RC):
        n_cols = len(methods) + 1  # one extrap subplot per method + bar chart
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4), squeeze=False)
        axes_row = axes[0]
        obs0 = observables[0]

        for i, m in enumerate(methods):
            draw_zne_extrapolation_on_ax(
                axes_row[i], results, gains, obs0,
                ideal_expectations, noisy_expectations, m,
            )
        add_zne_extrapolation_legend_to_first_axis(axes_row[0], results)

        ax_bar = axes_row[len(methods)]

        # Bar chart: x = fitting methods, bars = Mean, Median, Truth, ...
        obs0 = observables[0]
        param_names = ordered_names(list(results))
        n_params = len(param_names)
        n_meth = max(1, len(methods))
        bar_width = 0.8 / n_params
        x_base = np.arange(n_meth)
        ideal_val = ideal_expectations.get(obs0, np.nan)
        noisy_val = noisy_expectations.get(obs0, np.nan)
        if not np.isnan(ideal_val):
            ax_bar.axhline(ideal_val, label="Ideal", **IDEAL_STYLE)
        if not np.isnan(noisy_val):
            ax_bar.axhline(noisy_val, label="Noisy", **NOISY_STYLE)
        all_vals = []
        for i, pn in enumerate(param_names):
            vals = []
            for m in methods:
                r = results.get(pn, {}).get(obs0, {})
                extr = r.get("extrapolations", {})
                v = extr.get(m, {}).get("value", r.get("extrapolated", np.nan)) if extr else r.get("extrapolated", np.nan)
                v = v if not np.isnan(v) else 0
                vals.append(v)
                all_vals.append(v)
            offset = (i - (n_params - 1) / 2) * bar_width
            color = fill(pn)
            ax_bar.bar(
                x_base + offset, vals, bar_width,
                label=LABEL.get(pn, pn.capitalize()), color=color,
                edgecolor="#1E293B", linewidth=1,
            )
        y_min = min(all_vals + ([ideal_val] if not np.isnan(ideal_val) else []) + ([noisy_val] if not np.isnan(noisy_val) else []))
        y_max = max(all_vals + ([ideal_val] if not np.isnan(ideal_val) else []) + ([noisy_val] if not np.isnan(noisy_val) else []))
        y_span = y_max - y_min
        if y_span < 1e-8:
            y_span = 0.1
        margin = y_span * 0.15
        ax_bar.set_ylim(y_min - margin, y_max + margin)
        ax_bar.set_xticks(x_base)
        ax_bar.set_xticklabels([_METHOD_SHORT.get(m, m.capitalize()) for m in methods])
        ax_bar.yaxis.set_major_locator(mtick.MaxNLocator(4))
        ax_bar.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax_bar.grid(False)

        fig.supylabel("Expectation")
        fig.tight_layout()
        path = os.path.join(output_dir, out_filename)
        fig.savefig(path)
        plt.close(fig)
    return path


def load_and_plot(results_path: str, output_dir: Optional[str] = None) -> str:
    """
    Load ZNE results JSON and plot extrapolation.
    """
    with open(results_path) as f:
        data = json.load(f)
    out_dir = output_dir or os.path.dirname(results_path)
    gains = data.get("gains", [1.0, 1.2, 1.6])
    results = data.get("results", {})
    observables = data.get("observables", [])
    ideal = data.get("ideal_expectations", {})
    noisy = data.get("noisy_expectations", {})
    methods = data.get("extrapolation_methods", ["exponential"])
    path = plot_zne_extrapolation(
        results, gains, out_dir,
        observables=observables,
        ideal_expectations=ideal,
        noisy_expectations=noisy,
        extrapolation_methods=methods,
    )
    print(f"[ZNE] Plot saved → {path}")
    return path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m sbi_pec_zne.zne.visualize <zne_results.json> [output_dir]")
        sys.exit(1)
    res_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else None
    load_and_plot(res_path, out_dir)
