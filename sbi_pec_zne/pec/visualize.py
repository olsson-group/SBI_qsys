"""PEC visualization: mitigated ⟨O⟩ vs sample size (convergence plot) and sampling overhead."""

import json
import os
from typing import Optional

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from sbi_pec_zne.style import (
    EDGE, IDEAL_STYLE, LABEL, MARKER, NOISY_STYLE, VIZ_RC,
    fill, markersize, ordered_names,
)


def _legend_ordered(ax) -> None:
    """Two-column legend: parameters (truth, median, mean) on the left, Ideal / Noisy on the right."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    by_label = dict(zip(labels, handles))
    params = [LABEL[n] for n in ("truth", "median", "mean")]
    left = [lb for lb in params if lb in by_label]
    left += [lb for lb in labels if lb not in params and lb not in ("Ideal", "Noisy")]
    right = [lb for lb in ("Ideal", "Noisy") if lb in by_label]
    dummy = mlines.Line2D([], [], color="none", label="")
    right_handles = [by_label[lb] for lb in right] + [dummy] * (len(left) - len(right))
    right_labels = right + [""] * (len(left) - len(right))
    ax.legend([by_label[lb] for lb in left] + right_handles, left + right_labels,
              loc="best", frameon=False, ncol=2, columnspacing=0.8)


def _fmt_scaled(v: float, scale: float) -> str:
    """Format a scaled tick value without trailing zeros."""
    s = v / scale
    if abs(s - round(s)) < 1e-9:
        return f"{int(round(s))}"
    return f"{s:g}"


def draw_pec_single_observation_on_ax(
    ax,
    obs: str,
    results: dict,
    checkpoints: list,
    param_names: list,
    ideal_expectations: Optional[dict],
    noisy_expectations: Optional[dict],
    *,
    x_idx: list,
    x_scale: float,
    x_label: str,
    show_legend: bool = True,
) -> None:
    """Draw one PEC convergence panel (single observable) onto ``ax``."""
    z = 1
    if ideal_expectations and obs in ideal_expectations:
        ax.axhline(ideal_expectations[obs], label="Ideal", zorder=z, **IDEAL_STYLE)
        z += 1
    if noisy_expectations and obs in noisy_expectations:
        ax.axhline(noisy_expectations[obs], label="Noisy", zorder=z, **NOISY_STYLE)
        z += 1
    for pn in ordered_names(param_names):
        if obs not in results[pn]:
            continue
        vals = results[pn][obs]
        face = fill(pn)
        ax.plot(x_idx, vals, marker="None", linestyle="-", color=face, linewidth=2.2, alpha=0.6, zorder=z)
        ax.plot(
            x_idx, vals, marker=MARKER.get(pn, "o"), linestyle="None",
            markersize=markersize(pn), markerfacecolor=face,
            markeredgecolor=EDGE.get(pn, "#333333"), markeredgewidth=0.9,
            label=pn.capitalize(), zorder=z,
        )
        z += 1

    ax.set_xlabel(x_label)
    ax.set_ylabel("Expectation")
    if show_legend:
        _legend_ordered(ax)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xticks(x_idx)
    ax.set_xticklabels([_fmt_scaled(c, x_scale) for c in checkpoints])


def draw_pec_overhead_on_ax(ax, param_names: list, pec_costs: dict) -> None:
    """Draw sampling-overhead bar panel onto ``ax``."""
    if not pec_costs:
        ax.set_visible(False)
        return
    names = [pn for pn in ordered_names(list(pec_costs)) if pn in pec_costs]
    gammas = [pec_costs[pn] for pn in names]
    colors = [fill(pn) for pn in names]
    x = np.arange(len(names))

    ax.bar(
        x, gammas, color=colors, width=0.55,
        edgecolor="black", linewidth=0.6,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([n.capitalize() for n in names], rotation=0, ha="center")
    ax.set_ylabel("Sampling overhead")
    g_max = max(gammas)
    ax.set_ylim(0, g_max * 1.1)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4, integer=False))
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:g}")
    )


def plot_pec_convergence(
    results: dict,
    checkpoints: list,
    output_dir: str,
    observables: Optional[list] = None,
    ideal_expectations: Optional[dict] = None,
    noisy_expectations: Optional[dict] = None,
    pec_costs: Optional[dict] = None,
    out_filename: str = "pec_convergence.pdf",
) -> str:
    """
    Plot mitigated ⟨O⟩ vs PEC sample size (checkpoint) for each param summary,
    plus a sampling overhead subplot showing γ_total per param set.

    Parameters
    ----------
    results : dict
        Mapping param_name (mean, median, etc.) → {observable → list of values}.
    checkpoints : list
        Sample sizes (e.g. [20, 50, 100, 200]).
    output_dir : str
        Directory to save the figure.
    observables : list, optional
        Subset of observables to plot. None = all.
    ideal_expectations : dict, optional
        Mapping observable → ideal (noiseless) ⟨O⟩ for horizontal reference line.
    noisy_expectations : dict, optional
        Mapping observable → raw (unmitigated) noisy ⟨O⟩ for horizontal reference line.
    pec_costs : dict, optional
        Mapping param_name → γ_total (sampling overhead).
    out_filename : str
        Output filename (default: pec_convergence.pdf).

    Returns
    -------
    str
        Path to saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    obs_list = observables or list(next(iter(results.values())).keys())
    param_names = list(results.keys())
    n_obs = len(obs_list)

    # one convergence panel per observable plus one overhead panel
    n_cols = n_obs + 1
    with plt.rc_context(VIZ_RC):
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
        if n_cols == 1:
            axes = [axes]

        x_idx = list(range(len(checkpoints)))  # equally spaced indices
        ckpt_abs_max = max(abs(float(c)) for c in checkpoints) if checkpoints else 0.0
        x_exp = int(np.floor(np.log10(ckpt_abs_max))) if ckpt_abs_max >= 1000 else 0
        x_scale = 10 ** x_exp
        x_label = (
            "Random circuit instances"
            if x_exp == 0
            else rf"Random circuit instances [$\times 10^{x_exp}$]"
        )
        for ax_idx, obs in enumerate(obs_list):
            ax = axes[ax_idx]
            draw_pec_single_observation_on_ax(
                ax, obs, results, checkpoints, param_names,
                ideal_expectations, noisy_expectations,
                x_idx=x_idx, x_scale=x_scale, x_label=x_label,
            )

        ax_oh = axes[n_obs]
        if pec_costs:
            draw_pec_overhead_on_ax(ax_oh, param_names, pec_costs)
        else:
            ax_oh.set_visible(False)

        fig.tight_layout()
        path = os.path.join(output_dir, out_filename)
        fig.savefig(path)
        plt.close(fig)
    return path


def load_and_plot(results_path: str, output_dir: Optional[str] = None) -> str:
    """
    Load pec_results.json and plot convergence. Saves to same directory by default.

    Parameters
    ----------
    results_path : str
        Path to pec_results.json.
    output_dir : str, optional
        Where to save the figure. Default: same directory as results_path.

    Returns
    -------
    str
        Path to saved figure.
    """
    with open(results_path) as f:
        data = json.load(f)

    results = data["results"]
    checkpoints = data["checkpoints"]
    ideal_expectations = data.get("ideal_expectations")
    noisy_expectations = data.get("noisy_expectations")
    pec_costs = data.get("pec_costs")
    out_dir = output_dir or os.path.dirname(results_path)

    path = plot_pec_convergence(
        results, checkpoints, out_dir,
        ideal_expectations=ideal_expectations,
        noisy_expectations=noisy_expectations,
        pec_costs=pec_costs,
    )
    print(f"PEC convergence plot saved → {path}")
    return path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m sbi_pec_zne.pec.visualize <pec_results.json> [output_dir]")
        sys.exit(1)
    res_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else None
    load_and_plot(res_path, out_dir)
