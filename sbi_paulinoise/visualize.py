"""Visualisation: training loss, posterior boxplot, SNPE scaling law."""

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


def _link(path: str) -> str:
    """OSC 8 hyperlink — clickable in iTerm2, JupyterLab, VS Code terminal."""
    p = os.path.abspath(path)
    return f"\033]8;;file://{p}\033\\{p}\033]8;;\033\\"


def _infer_n_qubits(num_params: int) -> int | None:
    """
    Infer n_qubits from the parameter layout num_params = 15 * (n_qubits - 1).
    """
    if num_params % 15 == 0:
        n_cnots = num_params // 15
        return int(n_cnots + 1)
    return None


def _gauge_free_keep_mask_from_posterior(samples_all: np.ndarray) -> np.ndarray:
    """
    Match paulinoise_scatter_inference.pdf's default gauge-drop rule:
    drop the top-k parameters by posterior std, where k=6*(n_qubits-2).
    """
    std = samples_all.std(axis=0)
    num_params = std.size
    n_qubits = _infer_n_qubits(num_params)
    if n_qubits is None:
        # Fallback: keep everything if we can't infer.
        return np.ones((num_params,), dtype=bool)
    k = int(6 * (n_qubits - 2))
    k = max(0, min(k, num_params))
    keep = np.ones((num_params,), dtype=bool)
    if k == 0:
        return keep
    idx_drop = np.argsort(-std)[:k]
    keep[idx_drop] = False
    return keep


# ── Shared RC params ──────────────────────────────────────────────────────────

_VIZ_RC = {
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size":          16,
    "axes.labelsize":     17,
    "axes.titlesize":     17,
    "xtick.labelsize":    11,
    "ytick.labelsize":    14,
    "legend.fontsize":    14,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.major.size":   3.5,
    "ytick.major.size":   3.5,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.top":          False,
    "ytick.right":        False,
    "legend.frameon":     False,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
}


# ── Training loss ─────────────────────────────────────────────────────────────

def plot_training_loss(
    inference_obj,
    method_name: str,
    output_dir: str | None = None,
    save_plot: bool = True,
) -> None:
    """Print loss summary and optionally save loss curve to output_dir."""
    print("\n[Training Complete]")
    if not (hasattr(inference_obj, "_summary") and inference_obj._summary):
        print("  (Training summary not available)")
        return
    summary = inference_obj._summary
    train_losses = summary.get("training_log", [])
    val_losses   = summary.get("validation_log", [])
    if not (train_losses and val_losses):
        print("  (Training log not available in summary)")
        return
    print(f"  Final Train Loss: {train_losses[-1]:.6f}")
    print(f"  Final Val Loss:   {val_losses[-1]:.6f}")
    print(f"  Best Val Loss:    {min(val_losses):.6f}")
    print(f"  Total Epochs:     {len(train_losses)}")
    if not save_plot:
        return
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, "b-", label="Train", alpha=0.7)
    plt.plot(range(1, len(val_losses) + 1),   val_losses,   "r-", label="Val",   alpha=0.7)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"{method_name} Training Loss")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    path = os.path.join(output_dir, "paulinoise_training_loss.png") if output_dir else "paulinoise_training_loss.png"
    plt.savefig(path, dpi=150)
    print(f"  Training loss curve saved to {_link(path)}")
    plt.close()


# ── Posterior box plot ────────────────────────────────────────────────────────

def _make_boxplot(
    ax,
    data: np.ndarray,
    n_cols: int,
    *,
    flier_markersize: float = 2,
    line_scale: float = 1.0,
    showfliers: bool = True,
) -> None:
    flier_kw = (
        dict(
            marker=".",
            markersize=flier_markersize,
            alpha=0.35,
            markerfacecolor="#9DDDD5",
            markeredgecolor="none",
        )
        if showfliers
        else None
    )
    ax.boxplot(
        [data[:, j] for j in range(n_cols)],
        positions=range(n_cols),
        widths=0.55,
        patch_artist=True,
        showfliers=showfliers,
        flierprops=flier_kw,
        medianprops=dict(color="black", linewidth=0.8 * line_scale),
        boxprops=dict(facecolor="#9DDDD5", edgecolor="black", linewidth=0.8 * line_scale),
        whiskerprops=dict(color="black", linewidth=0.8 * line_scale, linestyle="-"),
        capprops=dict(color="black", linewidth=0.8 * line_scale),
    )


def _style_ax(ax, n_cols: int):
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(fmt)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5, zorder=1)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.tick_params(top=False, right=False)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    return fmt


_GATE_COLORS = [
    "#eaf4fb", "#fef9e7", "#eafaf1", "#fdf2f8",
    "#fef5e4", "#f4ecf7", "#e8f8f5", "#fdedec",
]


def plot_box(
    posterior_samples_np: np.ndarray,
    num_params: int,
    cx_topology: str,
    results_dir: str,
    truth_np: np.ndarray | None = None,
    cnot_indices: list[int] | None = None,
    out_filename: str = "paulinoise_box.pdf",
    font_scale: float = 1.0,
    fig_height: float = 4.0,
    fig_width: float | None = None,
    marker_scale: float = 1.0,
    truth_marker: str = "x",
    cnot_label_pad: float = 0.20,
    legend_outside: bool = False,
    compact_style: bool = False,
    max_cnots_per_row: int | None = None,
    showfliers: bool = True,
) -> None:
    """Save posterior boxplot of the gate parameters to results_dir/<out_filename>.

    cnot_indices:      subset of CNOT indices (0-based) to plot. None = all.
    font_scale:        multiply all font sizes by this factor (default 1.0 = unchanged).
    fig_height:        figure height *per row* in inches (default 4.0).
    marker_scale:      multiply ground-truth marker size by this factor (default 1.0).
    truth_marker:      matplotlib marker style for ground-truth overlay (default "x").
    cnot_label_pad:    vertical offset of CNOT labels below x-axis in axes coords (default 0.20).
    legend_outside:    if True, place legend as one horizontal row above the top axes.
    compact_style:     paper-friendly layout (short height, outside legend, narrow columns).
                       Also enabled automatically when cnot_indices is set.
    max_cnots_per_row: maximum CNOT gates per subplot row.  When the total number of
                       selected CNOTs exceeds this value the figure is split into
                       vertically stacked rows so it fits in a paper column.
                       ``None`` (default) = auto: 10 when n_sel > 10, else single row.
    """

    # ── Detect parameter structure ────────────────────────────────────────────
    if num_params % 15 != 0:
        raise ValueError(
            f"Cannot infer parameter structure from num_params={num_params}. Expected n_cnots*15."
        )
    n_cnots = num_params // 15
    n_qubits = n_cnots + 1
    print(f"Parameter structure: {num_params} gate params ({n_cnots} CNOTs, {n_qubits} qubits)")

    # ── Select subset of CNOTs ────────────────────────────────────────────────
    all_cnot_indices = list(range(n_cnots))
    sel = list(cnot_indices) if cnot_indices is not None else all_cnot_indices
    if cnot_indices is not None and sel:
        if max(sel) == n_cnots:
            sel = [i - 1 for i in sel]
        if min(sel) < 0 or max(sel) >= n_cnots:
            raise ValueError(f"cnot_indices out of range for n_cnots={n_cnots}: {cnot_indices}")
    cx_topology_norm = str(cx_topology).lower()

    if cx_topology_norm == "brickwall" and cnot_indices is None:
        even = [i for i in range(n_cnots) if i % 2 == 0]
        odd = [i for i in range(n_cnots) if i % 2 == 1]
        brickwall_order_all = even + odd
        sel = [i for i in brickwall_order_all if i in set(sel)]

    # Pauli ordering within each CNOT group (weight-1 first, weight-2 second)
    bases = ["I", "X", "Y", "Z"]
    pauli_2q_raw = [p1 + p2 for p1 in bases for p2 in bases if not (p1 == "I" and p2 == "I")]
    w1 = [p for p in pauli_2q_raw if "I" in p]
    w2 = [p for p in pauli_2q_raw if "I" not in p]
    pauli_2q = w1 + w2
    old_pos   = {lbl: idx for idx, lbl in enumerate(pauli_2q_raw)}
    perm_in_g = [old_pos[lbl] for lbl in pauli_2q]
    col_perm  = [i * 15 + j for i in sel for j in perm_in_g]

    cnot_pairs = [(i, i + 1) for i in sel]
    gate_labels = [f"$\\mathrm{{{lbl}}}$" for _ in sel for lbl in pauli_2q]
    n_sel = len(sel)
    n_sel_gate_params = n_sel * 15

    gate_data = posterior_samples_np[:, col_perm]

    # Compact, paper-friendly canvas
    use_compact = compact_style or cnot_indices is not None
    if use_compact:
        if fig_height == 4.0:
            fig_height = 2.3
        if font_scale == 1.0:
            font_scale = 0.9
        if legend_outside is False:
            legend_outside = True

    # ── Decide whether to use multi-row layout ───────────────────────────────
    if max_cnots_per_row is None:
        max_cnots_per_row = 10 if n_sel > 10 else n_sel
    use_multirow = n_sel > max_cnots_per_row

    if use_multirow:
        _plot_box_multirow(
            gate_data, n_sel, cnot_pairs, gate_labels, pauli_2q,
            col_perm, truth_np, results_dir, out_filename,
            font_scale=font_scale, fig_height=fig_height, fig_width=fig_width,
            marker_scale=marker_scale, truth_marker=truth_marker,
            cnot_label_pad=cnot_label_pad, legend_outside=legend_outside,
            use_compact=use_compact, max_cnots_per_row=max_cnots_per_row,
            showfliers=showfliers,
        )
        return

    # ── Single-row layout (original path) ────────────────────────────────────
    _font_keys = ("font.size", "axes.labelsize", "axes.titlesize",
                  "xtick.labelsize", "ytick.labelsize", "legend.fontsize")
    _rc = {**_VIZ_RC, **{k: _VIZ_RC[k] * font_scale for k in _font_keys if k in _VIZ_RC}}
    with plt.rc_context(_rc):
        if fig_width is not None:
            fw = fig_width
        else:
            per_col = 0.16 if use_compact else 0.32
            fw = max(10.0, n_sel_gate_params * per_col)
        fig, ax_g = plt.subplots(figsize=(fw, fig_height))

        # ── Gate panel ────────────────────────────────────────────────────────
        for i, (ctl, tgt) in enumerate(cnot_pairs):
            lo, hi = i * 15 - 0.5, i * 15 + 14.5
            ax_g.axvspan(lo, hi, color=_GATE_COLORS[i % len(_GATE_COLORS)], alpha=0.45, zorder=0)

        _make_boxplot(
            ax_g,
            gate_data,
            n_sel_gate_params,
            flier_markersize=1.2 if use_compact else 2,
            line_scale=0.75 if use_compact else 1.0,
            showfliers=showfliers,
        )
        posterior_handle = mpatches.Patch(
            facecolor="#9DDDD5", edgecolor="black", linewidth=0.8, label="Posterior samples"
        )
        truth_handle = None
        if truth_np is not None:
            gate_truth = truth_np[col_perm]
            ax_g.plot(range(n_sel_gate_params), gate_truth,
                      truth_marker, color="#E8000D", markersize=6 * marker_scale,
                      markeredgewidth=1.2 * marker_scale, linestyle="none", zorder=10)
            truth_handle = Line2D(
                [], [], marker=truth_marker, color="#E8000D", linestyle="none",
                markersize=6 * marker_scale, markeredgewidth=1.2 * marker_scale,
                label="Ground truth",
            )
            print("Visualization: Ground Truth overlaid.")
        else:
            print("Visualization: Ground Truth skipped.")

        _place_legend(ax_g, posterior_handle, truth_handle, legend_outside)
        fmt_g = _style_ax(ax_g, n_sel_gate_params)
        ax_g.set_ylabel("Posterior value")
        ax_g.set_xticks(range(n_sel_gate_params))
        ax_g.set_xticklabels(_split_pauli_labels(gate_labels), rotation=0, ha="center")
        ax_g.tick_params(axis="both", direction="out")
        fig.canvas.draw()
        off = fmt_g.get_offset()
        if off:
            ax_g.yaxis.offsetText.set_visible(True)
            ax_g.yaxis.offsetText.set_text(off)

        fig.tight_layout()
        fig.canvas.draw()

        _draw_cnot_labels(fig, ax_g, cnot_pairs, cnot_label_pad, font_scale)
        path = os.path.join(results_dir, out_filename)
        fig.savefig(path)
        plt.close(fig)
        print(f"Posterior boxplot saved to {_link(path)}")


def _split_pauli_labels(labels):
    """Turn '$\\mathrm{XY}$' → '$\\mathrm{X}$\\n$\\mathrm{Y}$' for two-line ticks."""
    return [f"$\\mathrm{{{lbl[0]}}}$\n$\\mathrm{{{lbl[1]}}}$"
            for lbl in [l.replace("$\\mathrm{", "").replace("}$", "") for l in labels]]


def _place_legend(ax, posterior_handle, truth_handle, legend_outside: bool):
    """Place Posterior/Ground-truth legend on *ax*."""
    if truth_handle is None:
        handles = [posterior_handle]
        kw = dict(ncol=1, frameon=not legend_outside, borderaxespad=0)
        if legend_outside:
            kw.update(loc="lower center", bbox_to_anchor=(0.5, 1.01))
        else:
            kw.update(loc="upper right", facecolor="white", framealpha=0.85, edgecolor="none")
        leg = ax.legend(handles=handles, **kw)
        leg.set_zorder(20)
    else:
        if legend_outside:
            leg_post = ax.legend(
                handles=[posterior_handle], loc="lower center",
                bbox_to_anchor=(0.42, 1.01), ncol=1, frameon=False, borderaxespad=0)
            ax.add_artist(leg_post)
            leg_truth = ax.legend(
                handles=[truth_handle], loc="lower center",
                bbox_to_anchor=(0.66, 1.01), ncol=1, frameon=False,
                borderaxespad=0, handletextpad=0.25)
        else:
            leg_post = ax.legend(
                handles=[posterior_handle], loc="upper right",
                frameon=True, facecolor="white", framealpha=0.85, edgecolor="none")
            ax.add_artist(leg_post)
            leg_truth = ax.legend(
                handles=[truth_handle], loc="upper right",
                bbox_to_anchor=(1.0, 0.90), frameon=True, facecolor="white",
                framealpha=0.85, edgecolor="none", handletextpad=0.25)
        leg_post.set_zorder(20)
        leg_truth.set_zorder(20)


def _draw_cnot_labels(fig, ax, cnot_pairs, cnot_label_pad, font_scale,
                      local_offset: int = 0):
    """Draw CNOT_{c→t} labels below an axes after tight_layout."""
    renderer = fig.canvas.get_renderer()
    ax_b = ax.transAxes.transform((0, 0))[1]
    ax_t = ax.transAxes.transform((0, 1))[1]
    ax_h = ax_t - ax_b
    tick_bottom_ax = 0.0
    if ax_h > 0:
        for lbl in ax.get_xticklabels():
            bb = lbl.get_window_extent(renderer)
            y_ax = (bb.y0 - ax_b) / ax_h
            if y_ax < tick_bottom_ax:
                tick_bottom_ax = y_ax
    cnot_y = min(-cnot_label_pad, tick_bottom_ax - 0.08)
    for i, (ctl, tgt) in enumerate(cnot_pairs):
        lo = (local_offset + i) * 15 - 0.5
        hi = lo + 15.0
        ax.text(
            (lo + hi) / 2, cnot_y,
            f"$\\mathrm{{CNOT}}_{{{ctl+1}\\rightarrow{tgt+1}}}$",
            ha="center", va="top",
            transform=ax.get_xaxis_transform(),
            fontsize=14 * font_scale, color="black",
        )


def _plot_box_multirow(
    gate_data: np.ndarray,
    n_sel: int,
    cnot_pairs: list[tuple[int, int]],
    gate_labels: list[str],
    pauli_2q: list[str],
    col_perm: list[int],
    truth_np: np.ndarray | None,
    results_dir: str,
    out_filename: str,
    *,
    font_scale: float,
    fig_height: float,
    fig_width: float | None,
    marker_scale: float,
    truth_marker: str,
    cnot_label_pad: float,
    legend_outside: bool,
    use_compact: bool,
    max_cnots_per_row: int,
    showfliers: bool = True,
) -> None:
    """Render the boxplot split into multiple vertically stacked rows."""
    import math
    n_rows = math.ceil(n_sel / max_cnots_per_row)

    # Split CNOTs into row-sized chunks
    row_chunks: list[list[int]] = []
    for r in range(n_rows):
        start = r * max_cnots_per_row
        end = min(start + max_cnots_per_row, n_sel)
        row_chunks.append(list(range(start, end)))

    max_cols = max_cnots_per_row * 15
    per_col = 0.16 if use_compact else 0.25
    fw = fig_width if fig_width is not None else max(7.0, max_cols * per_col)
    row_h = fig_height if fig_height != 4.0 else 2.0

    _font_keys = ("font.size", "axes.labelsize", "axes.titlesize",
                  "xtick.labelsize", "ytick.labelsize", "legend.fontsize")
    _rc = {**_VIZ_RC, **{k: _VIZ_RC[k] * font_scale for k in _font_keys if k in _VIZ_RC}}

    with plt.rc_context(_rc):
        fig, axes = plt.subplots(
            n_rows, 1, figsize=(fw, row_h * n_rows),
            squeeze=False,
        )
        axes = [ax for ax in axes[:, 0]]

        # Pre-compute shared y-limits across all data for visual consistency.
        y_min_global = float(gate_data.min())
        y_max_global = float(gate_data.max())
        y_margin = (y_max_global - y_min_global) * 0.08
        y_lo = y_min_global - y_margin
        y_hi = y_max_global + y_margin
        if truth_np is not None:
            gate_truth_all = truth_np[col_perm]
            y_lo = min(y_lo, float(gate_truth_all.min()) - y_margin)
            y_hi = max(y_hi, float(gate_truth_all.max()) + y_margin)

        truth_printed = False
        for row_idx, chunk_indices in enumerate(row_chunks):
            ax = axes[row_idx]
            n_cnots_row = len(chunk_indices)
            n_cols_row = n_cnots_row * 15

            # Slice the data columns for this row
            col_start = chunk_indices[0] * 15
            col_end = col_start + n_cols_row
            row_data = gate_data[:, col_start:col_end]
            row_pairs = cnot_pairs[chunk_indices[0]:chunk_indices[0] + n_cnots_row]
            row_labels = gate_labels[col_start:col_end]

            # Alternating background bands
            for i in range(n_cnots_row):
                lo = i * 15 - 0.5
                hi = lo + 15.0
                color_idx = (chunk_indices[i]) % len(_GATE_COLORS)
                ax.axvspan(lo, hi, color=_GATE_COLORS[color_idx], alpha=0.45, zorder=0)

            _make_boxplot(
                ax, row_data, n_cols_row,
                flier_markersize=1.2 if use_compact else 2,
                line_scale=0.75 if use_compact else 1.0,
                showfliers=showfliers,
            )

            # Ground truth overlay
            if truth_np is not None:
                row_truth = gate_truth_all[col_start:col_end]
                ax.plot(range(n_cols_row), row_truth,
                        truth_marker, color="#E8000D", markersize=6 * marker_scale,
                        markeredgewidth=1.2 * marker_scale, linestyle="none", zorder=10)
                if not truth_printed:
                    print("Visualization: Ground Truth overlaid.")
                    truth_printed = True

            # Legend only on the first row
            if row_idx == 0:
                posterior_handle = mpatches.Patch(
                    facecolor="#9DDDD5", edgecolor="black", linewidth=0.8, label="Posterior samples")
                truth_handle = None
                if truth_np is not None:
                    truth_handle = Line2D(
                        [], [], marker=truth_marker, color="#E8000D", linestyle="none",
                        markersize=6 * marker_scale, markeredgewidth=1.2 * marker_scale,
                        label="Ground truth")
                _place_legend(ax, posterior_handle, truth_handle, legend_outside=True)

            if truth_np is None and row_idx == 0 and not truth_printed:
                print("Visualization: Ground Truth skipped.")

            _style_ax(ax, n_cols_row)
            ax.set_ylim(y_lo, y_hi)
            ax.set_ylabel("Posterior value")
            ax.set_xticks(range(n_cols_row))
            ax.set_xticklabels(_split_pauli_labels(row_labels), rotation=0, ha="center")
            ax.tick_params(axis="both", direction="out")

        fig.tight_layout(h_pad=1.8)
        fig.canvas.draw()

        # CNOT labels for each row (must be after tight_layout)
        for row_idx, chunk_indices in enumerate(row_chunks):
            ax = axes[row_idx]
            n_cnots_row = len(chunk_indices)
            row_pairs = cnot_pairs[chunk_indices[0]:chunk_indices[0] + n_cnots_row]
            _draw_cnot_labels(fig, ax, row_pairs, cnot_label_pad, font_scale)

        path = os.path.join(results_dir, out_filename)
        fig.savefig(path)
        plt.close(fig)
        print(f"Posterior boxplot saved to {_link(path)}")


# ── Wrapped (multi-row) posterior box plot ────────────────────────────────────

def plot_box_wrapped(
    posterior_samples_np: np.ndarray,
    num_params: int,
    cx_topology: str,
    results_dir: str,
    truth_np: np.ndarray | None = None,
    out_filename: str = "paulinoise_box_wrapped.pdf",
    cnots_per_row: int = 5,
    row_height: float = 0.85,
    fig_width: float = 8.5,
    marker_scale: float = 1.0,
    truth_marker: str = "x",
) -> None:
    """Publication-quality multi-row wrapped boxplot.

    The CNOT groups are split into rows of `cnots_per_row` each.
    For brickwall topology, even/odd layers are visually separated.
    """
    # ── Detect parameter structure ────────────────────────────────────────────
    if num_params % 15 != 0:
        raise ValueError(f"Cannot infer parameter structure from num_params={num_params}.")
    n_cnots = num_params // 15

    cx_topology_norm = str(cx_topology).lower()

    # Pauli ordering (weight-1 first, weight-2 second)
    bases = ["I", "X", "Y", "Z"]
    pauli_2q_raw = [p1 + p2 for p1 in bases for p2 in bases if not (p1 == "I" and p2 == "I")]
    w1 = [p for p in pauli_2q_raw if "I" in p]
    w2 = [p for p in pauli_2q_raw if "I" not in p]
    pauli_2q = w1 + w2
    old_pos = {lbl: idx for idx, lbl in enumerate(pauli_2q_raw)}
    perm_in_g = [old_pos[lbl] for lbl in pauli_2q]

    # CNOT display order with proper layer separation
    is_brickwall = cx_topology_norm == "brickwall"
    if is_brickwall:
        even = [i for i in range(n_cnots) if i % 2 == 0]
        odd = [i for i in range(n_cnots) if i % 2 == 1]
        even_chunks = [even[i:i+cnots_per_row] for i in range(0, len(even), cnots_per_row)]
        odd_chunks = [odd[i:i+cnots_per_row] for i in range(0, len(odd), cnots_per_row)]
        row_chunks = even_chunks + odd_chunks
        n_even_rows = len(even_chunks)
        display_order = even + odd
    else:
        display_order = list(range(n_cnots))
        row_chunks = [display_order[i:i+cnots_per_row]
                      for i in range(0, len(display_order), cnots_per_row)]
        n_even_rows = len(row_chunks)

    n_rows = len(row_chunks)

    # ── Publication RC params ─────────────────────────────────────────────────
    _pub_rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 9,
        "axes.linewidth": 0.4,
        "xtick.major.width": 0.3,
        "ytick.major.width": 0.3,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }

    # Colors: teal for odd layer (bonds 0,2,4,...), purple for even layer (bonds 1,3,5,...)
    _box_odd = "#7ecac0"
    _box_odd_edge = "#3a9a90"
    _box_even = "#b8a9d4"
    _box_even_edge = "#7a68a6"

    with plt.rc_context(_pub_rc):
        # Extra vertical space between even/odd sections
        height_ratios = [1] * n_rows
        fig = plt.figure(figsize=(fig_width, row_height * n_rows + 0.5))
        gs = fig.add_gridspec(
            n_rows, 1,
            height_ratios=height_ratios,
            hspace=0.75,
        )
        axes_list = [fig.add_subplot(gs[i, 0]) for i in range(n_rows)]

        # Global y-limits for consistent scaling across rows
        all_col_perm = [i * 15 + j for i in display_order for j in perm_in_g]
        gate_data_all = posterior_samples_np[:, all_col_perm]
        y_lo = float(np.percentile(gate_data_all, 0.3))
        y_hi = float(np.percentile(gate_data_all, 99.7))
        y_pad = (y_hi - y_lo) * 0.08
        ylim = (y_lo - y_pad, y_hi + y_pad)

        full_row_cols = cnots_per_row * 15

        for row_idx in range(n_rows):
            ax = axes_list[row_idx]
            row_cnots = row_chunks[row_idx]
            n_sel = len(row_cnots)
            n_cols = n_sel * 15
            is_first_section = row_idx < n_even_rows

            col_perm = [i * 15 + j for i in row_cnots for j in perm_in_g]
            gate_data = posterior_samples_np[:, col_perm]

            # First section = bond indices 0,2,4,... = odd layer; second = even layer
            box_fc = _box_odd if is_first_section else _box_even
            box_ec = _box_odd_edge if is_first_section else _box_even_edge

            # Subtle alternating background
            for i in range(n_sel):
                lo, hi = i * 15 - 0.5, i * 15 + 14.5
                bg = "#f7f9fb" if i % 2 == 0 else "#fcfaf7"
                ax.axvspan(lo, hi, color=bg, alpha=1.0, zorder=0, linewidth=0)

            # Thin vertical separators between CNOT groups
            for i in range(1, n_sel):
                ax.axvline(i * 15 - 0.5, color="#dddddd", linewidth=0.25, zorder=1)

            # Boxplot
            ax.boxplot(
                [gate_data[:, j] for j in range(n_cols)],
                positions=range(n_cols),
                widths=0.55,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color="#2c3e50", linewidth=0.5),
                boxprops=dict(facecolor=box_fc, edgecolor=box_ec, linewidth=0.35, alpha=0.9),
                whiskerprops=dict(color=box_ec, linewidth=0.35, linestyle="-"),
                capprops=dict(color=box_ec, linewidth=0.35),
            )

            # Ground truth overlay
            if truth_np is not None:
                gate_truth = truth_np[col_perm]
                ax.plot(range(n_cols), gate_truth,
                        truth_marker, color="#c0392b", markersize=2.5 * marker_scale,
                        markeredgewidth=0.6 * marker_scale, linestyle="none", zorder=10)

            # Y-axis
            ax.set_ylim(ylim)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, symmetric=False))
            fmt = ticker.ScalarFormatter(useMathText=True)
            fmt.set_scientific(True)
            fmt.set_powerlimits((0, 0))
            ax.yaxis.set_major_formatter(fmt)
            # Only show ×10⁻⁴ offset on the first row
            if row_idx > 0:
                ax.yaxis.get_offset_text().set_visible(False)
            ax.grid(axis="y", alpha=0.25, linewidth=0.25, zorder=1, color="#aaaaaa")
            ax.set_xlim(-0.5, full_row_cols - 0.5)

            # X-axis: Pauli labels for each parameter
            pauli_tick_labels = [f"{lbl[0]}\n{lbl[1]}" for _ in row_cnots for lbl in pauli_2q]
            ax.set_xticks(range(n_cols))
            ax.set_xticklabels(pauli_tick_labels, fontsize=5.5, ha="center",
                               linespacing=0.8, color="#444444")
            ax.tick_params(axis="x", length=1.5, width=0.2, pad=0.5)
            ax.tick_params(axis="y", direction="out", pad=2, labelsize=8)

            # CNOT labels centered below each group
            for i, cnot_idx in enumerate(row_cnots):
                ctl, tgt = cnot_idx + 1, cnot_idx + 2
                center_x = i * 15 + 7
                ax.text(
                    center_x, -0.32,
                    f"$\\mathrm{{CX}}_{{{ctl},{tgt}}}$",
                    ha="center", va="top",
                    transform=ax.get_xaxis_transform(),
                    fontsize=9, color="#333333",
                )

            # Spines
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            ax.spines["bottom"].set_linewidth(0.3)
            ax.spines["left"].set_linewidth(0.3)

        # Shared y-axis label
        fig.text(0.008, 0.5, "Posterior value", va="center", rotation="vertical",
                 fontsize=14, color="#2c3e50")

        # Layer annotations removed — legend already distinguishes odd/even by color.

        # Legend centered at top
        odd_handle = mpatches.Patch(
            facecolor=_box_odd, edgecolor=_box_odd_edge, linewidth=0.5, label="Odd layer"
        )
        even_handle = mpatches.Patch(
            facecolor=_box_even, edgecolor=_box_even_edge, linewidth=0.5, label="Even layer"
        )
        handles = [odd_handle, even_handle]
        if truth_np is not None:
            truth_handle = Line2D(
                [], [], marker=truth_marker, color="#c0392b", linestyle="none",
                markersize=3.5 * marker_scale, markeredgewidth=0.6 * marker_scale,
                label="Ground truth",
            )
            handles.append(truth_handle)
        axes_list[0].legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(handles),
            frameon=False,
            borderaxespad=0,
            columnspacing=1.2,
            handletextpad=0.3,
            fontsize=9,
        )

        fig.subplots_adjust(left=0.055, right=0.995, top=0.955, bottom=0.025)
        path = os.path.join(results_dir, out_filename)
        fig.savefig(path)
        plt.close(fig)
        print(f"Wrapped boxplot saved to {_link(path)}")


# ── SNPE scaling law ──────────────────────────────────────────────────────────

def plot_scaling(
    nmae_all: list[float],
    nmae_gauge_free: list[float] | None,
    n_total: list[int],
    sims_per_round: int,
    results_dir: str,
    *,
    r2_all: list[float] | None = None,
    r2_gauge_free: list[float] | None = None,
    legend_outside: bool = True,
    legend_separate_pdf: bool = True,
    max_n_total: int | None = None,
    fig_width: float = 4.0,
    fig_height: float = 2.0,
) -> None:
    """Save nMAE vs round scaling law to results_dir/paulinoise_scaling.pdf."""
    _sl_rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        # Ensure mathtext uses a consistent, available font.
        "mathtext.fontset": "dejavusans",
        "font.size": 11, "axes.labelsize": 12, "legend.fontsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7,
        "ytick.major.width": 0.7, "xtick.major.size": 3.0,
        "ytick.major.size": 3.0, "xtick.direction": "out",
        "ytick.direction": "out", "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    }
    # Optionally truncate by cumulative training data (absolute count).
    if max_n_total is not None and len(n_total):
        keep_idx = [i for i, t in enumerate(n_total) if int(t) <= int(max_n_total)]
        if keep_idx:
            nmae_all = [nmae_all[i] for i in keep_idx]
            n_total = [n_total[i] for i in keep_idx]
            if nmae_gauge_free is not None:
                nmae_gauge_free = [nmae_gauge_free[i] for i in keep_idx]
            if r2_all is not None:
                r2_all = [r2_all[i] for i in keep_idx]
            if r2_gauge_free is not None:
                r2_gauge_free = [r2_gauge_free[i] for i in keep_idx]

    n_pts = len(nmae_all)
    # X axis: cumulative training data with a fixed unit ×10^4 (shared across experiments).
    # Use n_total (preferred) so sims_per_round differences don't change the x-scale.
    x_rounds = np.asarray(n_total, dtype=float) / 1e4
    xlabel = r"Cumulative training data [$\times 10^4$]"

    with plt.rc_context(_sl_rc):
        # Match the paper's small figure size (e.g. training_data_vs_qubits.pdf).
        # Use constrained layout to avoid tight_layout warnings with twin axes
        # and/or outside legends.
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.plot(x_rounds, nmae_all, "o-", color="#4DB8A8", linewidth=1.2, markersize=6,
                markerfacecolor="#9DDDD5", markeredgecolor="#2E8A80", markeredgewidth=0.8,
                label="nMAE (all)")
        if nmae_gauge_free is not None and len(nmae_gauge_free) == n_pts:
            ax.plot(
                x_rounds, nmae_gauge_free, "D-", color="#4DB8A8", linewidth=1.2, markersize=6,
                markerfacecolor="white", markeredgecolor="#2E8A80", markeredgewidth=0.8,
                label="nMAE (gauge-free)",
            )
        ax2 = None
        if r2_all is not None and len(r2_all) == n_pts:
            ax2 = ax.twinx()
            ax2.plot(
                x_rounds, r2_all,
                "s-", color="#8B65C8", linewidth=1.2, markersize=6,
                markerfacecolor="#C4A8E8", markeredgecolor="#6040A0", markeredgewidth=0.8,
                label=r"$R^2$ (all)",
            )
            if r2_gauge_free is not None and len(r2_gauge_free) == n_pts:
                ax2.plot(
                    x_rounds, r2_gauge_free,
                    "^-", color="#8B65C8", linewidth=1.2, markersize=6,
                    markerfacecolor="white", markeredgecolor="#6040A0", markeredgewidth=0.8,
                    label=r"$R^2$ (gauge-free)",
                )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("nMAE [%]")
        if ax2 is not None:
            ax2.set_ylabel(r"$R^2$")
            # Force math font on right-axis tick labels.
            ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: rf"${x:g}$"))
            ax2.tick_params(axis="y", direction="out", length=3.0, width=0.7)
        # Fixed x ticks (in units of ×10^4) for consistent comparisons.
        # Only keep ticks within the plotted range.
        fixed_xticks = np.array([0, 1, 2, 3, 4, 5], dtype=float)
        xmax = float(np.nanmax(x_rounds)) if len(x_rounds) else 0.0
        xt = fixed_xticks[fixed_xticks <= xmax + 1e-12]
        ax.set_xticks(xt.tolist())
        ax.set_xticklabels([str(int(v)) for v in xt.tolist()])
        ax.grid(False)

        # Use the scatter-plot standard for R^2 axis ticks/padding.
        if ax2 is not None:
            right_ticks = [0.1, 0.4, 0.7, 1.0]
            ax2.set_yticks(right_ticks)
            ax2.set_ylim(-0.1, 1.2)

            # Left axis: keep TRUE tick values, but force their HEIGHTS to align with right ticks.
            # Solve for left ylim so that left_ticks fall at the same fractional heights as right_ticks.
            left_ticks = np.array([0.0, 8.0, 16.0, 24.0], dtype=float)
            ax.set_yticks(left_ticks.tolist())

            y2min, y2max = ax2.get_ylim()
            rt = np.array(right_ticks, dtype=float)
            fracs = (rt - y2min) / (y2max - y2min)
            f1, f4 = float(fracs[0]), float(fracs[-1])
            if abs(f4 - f1) > 1e-12:
                d = (left_ticks[-1] - left_ticks[0]) / (f4 - f1)  # d = b-a
                a = left_ticks[0] - f1 * d
                b = a + d
                ax.set_ylim(float(a), float(b))
        else:
            # Fallback: single-axis plot.
            ax.set_yticks([0, 8, 16, 24])
            ax.set_ylim(-2, 26)

        handles, labels = ax.get_legend_handles_labels()
        if ax2 is not None:
            h2, l2 = ax2.get_legend_handles_labels()
            handles += h2
            labels += l2

        if legend_separate_pdf:
            # Save legend as its own 2×2 PDF and keep main plot clean.
            leg_fig = plt.figure(figsize=(4.6, 1.2))
            leg_fig.legend(
                handles,
                labels,
                frameon=False,
                loc="center",
                ncol=2,
                columnspacing=1.2,
                handlelength=2.2,
                handletextpad=0.6,
            )
            leg_fig.tight_layout(pad=0.1)
            leg_path = os.path.join(results_dir, "paulinoise_scaling_legend.pdf")
            leg_fig.savefig(leg_path)
            plt.close(leg_fig)
        else:
            # Legend on the main figure (2×2 layout).
            if legend_outside:
                ax.legend(
                    handles,
                    labels,
                    frameon=False,
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    borderaxespad=0.0,
                    ncol=2,
                )
            else:
                ax.legend(
                    handles,
                    labels,
                    frameon=False,
                    loc="center right",
                    ncol=2,
                )

        fig.tight_layout()
        path = os.path.join(results_dir, "paulinoise_scaling.pdf")
        fig.savefig(path)
        plt.close(fig)
    print(f"Scaling law saved to {_link(path)}")


# ── Loaders for saved results ──────────────────────────────────────────────────

def _load_config(results_dir: str) -> dict:
    """Parse config_snapshot.txt into a dict (key: value)."""
    cfg = {}
    path = os.path.join(results_dir, "config_snapshot.txt")
    with open(path) as f:
        for line in f:
            line = line.strip()
            if ": " in line:
                k, v = line.split(": ", 1)
                cfg[k.strip()] = v.strip()
    return cfg


def _load_posterior_samples(results_dir: str) -> np.ndarray:
    """Load posterior samples from paulinoise_inference_params.csv (rows = samples)."""
    path = os.path.join(results_dir, "paulinoise_inference_params.csv")
    return np.loadtxt(path, delimiter=",")


def _load_truth(results_dir: str) -> np.ndarray | None:
    """Load ground-truth parameters.

    Searches in order:
        1. THETA_TRUTH_FILE from config_snapshot.txt (relative to project root)
        2. <results_dir>/paulinoise_1_parameters.csv
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 1. From config_snapshot
    cfg = _load_config(results_dir)
    truth_file = cfg.get("THETA_TRUTH_FILE", "")
    if truth_file and truth_file != "None":
        path = truth_file if os.path.isabs(truth_file) else os.path.join(root, truth_file)
        if os.path.exists(path):
            return np.loadtxt(path, delimiter=",").flatten()

    # 2. Fallback: results dir
    path = os.path.join(results_dir, "paulinoise_1_parameters.csv")
    if os.path.exists(path):
        return np.loadtxt(path, delimiter=",").flatten()

    return None


def _load_posterior_mean_median_per_round(
    results_dir: str,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """
    Parse posterior_mean_median_per_round.txt.

    Returns two dicts keyed by round index:
        rounds_mean[r]   → 1-D array of per-parameter posterior means
        rounds_median[r] → 1-D array of per-parameter posterior medians
    """
    path = os.path.join(results_dir, "posterior_mean_median_per_round.txt")
    rounds_mean: dict[int, np.ndarray] = {}
    rounds_median: dict[int, np.ndarray] = {}
    current_round = -1
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            tag = parts[0]
            if tag == "round":
                current_round = int(parts[1])
            elif tag == "mean":
                rounds_mean[current_round] = np.array([float(x) for x in parts[1:]])
            elif tag == "median":
                rounds_median[current_round] = np.array([float(x) for x in parts[1:]])
    return rounds_mean, rounds_median


def _compute_nmae(results_dir: str, prior_high: float = 0.001, *, keep_mask: np.ndarray | None = None):
    """
    Compute normalised MAE (% of prior range) per round from the saved txt file.

    Returns (mae_mean, mae_median, round_ids) where each entry corresponds to
    one saved round (including round 0 = prior).
    """
    truth = _load_truth(results_dir)
    if truth is None:
        raise FileNotFoundError(
            f"paulinoise_1_parameters.csv not found in {results_dir}; "
            "ground truth required to compute nMAE."
        )
    rounds_mean, rounds_median = _load_posterior_mean_median_per_round(results_dir)
    round_ids = sorted(rounds_mean.keys())
    mae_mean, mae_median = [], []
    if keep_mask is not None:
        truth = truth[keep_mask]
    for r in round_ids:
        m = rounds_mean[r]
        med = rounds_median[r]
        if keep_mask is not None:
            m = m[keep_mask]
            med = med[keep_mask]
        mae_mean.append(np.mean(np.abs(m - truth)) / prior_high * 100)
        mae_median.append(np.mean(np.abs(med - truth)) / prior_high * 100)
    return mae_mean, mae_median, round_ids


def _load_r2_per_round(results_dir: str) -> dict[int, tuple[float, float]]:
    """
    Load per-round R^2 from r2_per_round.txt.

    Expected CSV columns:
        round,n_train,n_samples,r2_all,r2_no_gauge

    Returns:
        dict[round] -> (r2_all, r2_no_gauge)
    """
    path = os.path.join(results_dir, "r2_per_round.txt")
    if not os.path.exists(path):
        return {}
    r2: dict[int, tuple[float, float]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            try:
                r = int(parts[0])
                r2_all = float(parts[3])
                r2_no_gauge = float(parts[4])
            except ValueError:
                continue
            r2[r] = (r2_all, r2_no_gauge)
    return r2


# ── Scatter inference plot ────────────────────────────────────────────────────

def plot_scatter_inference(
    samples: np.ndarray,
    truth: np.ndarray,
    results_dir: str,
    prior_high: float = 0.001,
    out_filename: str = "paulinoise_scatter_inference.pdf",
    *,
    two_panel_drop_gauge: bool = True,
    drop_by: str = "std_topk_from_nqubits",
    drop_std_quantile: float = 0.8,
    drop_std_topk: int | None = None,
) -> None:
    """
    Scatter plot of posterior mean ± std vs true parameter.
    PRL single-column style.

    Parameters
    ----------
    samples    : (n_samples, num_params) posterior samples
    truth      : (num_params,) ground-truth parameters
    prior_high : upper bound of the prior (used to set axis ticks)
    out_filename : output filename inside results_dir
    """
    def _infer_n_qubits(num_params: int) -> int | None:
        """
        Infer n_qubits from the parameter layout num_params = 15 * (n_qubits - 1).
        """
        if num_params % 15 == 0:
            n_cnots = num_params // 15
            return int(n_cnots + 1)
        return None

    def _drop_mask_from_posterior(samples_all: np.ndarray) -> np.ndarray:
        """Drop 'unlearnable' parameters based on posterior spread (std)."""
        std = samples_all.std(axis=0)
        num_params = std.size

        if drop_by == "std_quantile":
            q = float(drop_std_quantile)
            if not (0.0 < q < 1.0):
                raise ValueError("drop_std_quantile must be in (0, 1).")
            thr = float(np.quantile(std, q))
            return std > thr

        if drop_by in ("std_topk", "std_topk_from_nqubits"):
            if drop_by == "std_topk_from_nqubits":
                n_qubits = _infer_n_qubits(num_params)
                if n_qubits is None:
                    raise ValueError(
                        f"Could not infer n_qubits from num_params={num_params}. "
                        "Set drop_by='std_topk' and provide drop_std_topk explicitly."
                    )
                k = int(6 * (n_qubits - 2))
            else:
                if drop_std_topk is None:
                    raise ValueError("drop_std_topk must be set when drop_by='std_topk'.")
                k = int(drop_std_topk)

            k = max(0, min(k, num_params))
            drop = np.zeros((num_params,), dtype=bool)
            if k == 0:
                return drop
            idx = np.argsort(-std)[:k]
            drop[idx] = True
            return drop

        raise ValueError(
            f"Unknown drop_by={drop_by!r} (expected 'std_quantile', 'std_topk', or 'std_topk_from_nqubits')."
        )

    # Match the scaling-plot look & size (fonts/linewidths/figsize).
    _sc_rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }

    # Shared exponent for tick labels (derived from prior_high).
    exp_tick = int(np.floor(np.log10(prior_high))) if prior_high > 0 else 0
    mult_str = rf"$\times10^{{{exp_tick}}}$"

    def _panel(ax, samples_p: np.ndarray, truth_p: np.ndarray, title: str, *, set_ylabel: bool = True):
        post_mean = samples_p.mean(axis=0)
        post_std  = samples_p.std(axis=0)

        # Axis limits: enforce strict [0, prior_high] so the scaled axis is exactly 0–1.
        lim = float(prior_high)
        ticks = [prior_high * k / 4 for k in range(1, 5)]

        errors   = np.abs(samples_p - truth_p[np.newaxis, :])
        mean_val = float(errors.mean())
        sd_val   = float(errors.std())
        exp_shared = int(np.floor(np.log10(abs(mean_val)))) if mean_val != 0 else 0
        scale      = 10 ** exp_shared
        m_mean     = mean_val / scale
        m_sd       = sd_val   / scale
        # R^2 for posterior mean as a predictor of truth
        y = post_mean
        ss_res = float(np.sum((truth_p - y) ** 2))
        ss_tot = float(np.sum((truth_p - float(np.mean(truth_p))) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        # Annotation (match previous style) + use a semi-transparent box like plot_box legend.
        annot = (
            rf"$\mathrm{{MAE\pm s.d.}}: ({m_mean:.1f}\pm{m_sd:.1f})\times10^{{{exp_shared}}}$"
            "\n"
            + rf"$R^2={r2:.3f}$"
        )

        tick_scale = 10 ** exp_tick
        tick_fmt   = ticker.FuncFormatter(lambda x, _: f"{x / tick_scale:g}" if x != 0 else "0")

        # Scatter styling: match the "Mean" marker style used in plot_scaling().
        # (filled circle with teal edge; light face color)
        ax.errorbar(
            truth_p, post_mean, yerr=post_std,
            fmt="o",
            color="#4DB8A8",
            markerfacecolor="#9DDDD5",
            markeredgecolor="#2E8A80",
            markeredgewidth=0.6,
            markersize=2.5,
            linewidth=0,
            elinewidth=0.6,
            ecolor="#4DB8A8",
            alpha=0.5,
            capsize=1.5,
            capthick=0.6,
        )
        ax.plot([0, lim], [0, lim], "k--", alpha=0.6, linewidth=0.9)
        # Keep the previous compact layout; only ensure the font *family* matches
        # the global style via the rc_context in reproduce().
        ax.text(
            0.04, 0.93, annot,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=9,
            linespacing=1.25,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, boxstyle="round,pad=0.25"),
            clip_on=True,
        )
        ax.set_title(title, pad=2)
        # Move the shared ×10^n scaling into the axis names (labels), rather than as separate annotations.
        ax.set_xlabel(rf"Truth [{mult_str}]")
        if set_ylabel:
            ax.set_ylabel(rf"Posterior [{mult_str}]")
        ax.tick_params(
            axis="both", which="both",
            direction="out", labelsize=9,
            length=4, width=0.9,
            top=False, right=False,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
        ax.xaxis.set_major_formatter(tick_fmt)
        ax.yaxis.set_major_formatter(tick_fmt)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        # scaling factor is already in axis labels

    if two_panel_drop_gauge:
        drop = _drop_mask_from_posterior(samples)
        keep = ~drop
        samples_keep = samples[:, keep]
        truth_keep = truth[keep]

        with plt.rc_context(_sc_rc):
            fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(5, 2), sharey=True)
            _panel(ax_l, samples, truth, f"All (n={samples.shape[1]})", set_ylabel=True)
            _panel(ax_r, samples_keep, truth_keep, f"Gauge-free (n={samples_keep.shape[1]})", set_ylabel=False)
            fig.tight_layout(pad=0.3, w_pad=0.6)
    else:
        with plt.rc_context(_sc_rc):
            fig, ax = plt.subplots(figsize=(5, 2))
            _panel(ax, samples, truth, f"All (n={samples.shape[1]})")
            fig.tight_layout(pad=0.3)

    p = os.path.join(results_dir, out_filename)
    fig.savefig(p)
    plt.close(fig)
    print(f"  Scatter inference saved → {_link(p)}")


# ── Reproduce from saved results ───────────────────────────────────────────────

def reproduce(
    results_dir: str,
    cnot_select: list[int] | None = None,
    *,
    font_scale: float = 1.0,
    fig_height: float = 4.0,
    fig_width: float | None = None,
    marker_scale: float = 1.0,
    truth_marker: str = "x",
    cnot_label_pad: float = 0.20,
    legend_outside: bool = False,
    compact_box: bool = False,
    scaling_legend_separate_pdf: bool = True,
    scaling_legend_outside: bool = True,
    scaling_max_n_total: int | None = None,
    scaling_fig_width: float = 4.0,
    scaling_fig_height: float = 2.0,
) -> None:
    """
    Reproduce paulinoise_box.pdf and paulinoise_scaling.pdf from a saved results directory.

    cnot_select:  list of 1-based ctrl-qubit numbers to plot in the given order,
                  e.g. [1, 3, 49, 2] for CNOT1→2, CNOT3→4, CNOT49→50, CNOT2→3.
                  Saves paulinoise_box_sel_<ids>.pdf.

    Reads:
        config_snapshot.txt              — experiment hyper-parameters
        paulinoise_inference_params.csv         — posterior samples (rows × params)
        paulinoise_1_parameters.csv             — ground-truth parameters (optional)
        posterior_mean_median_per_round.txt — per-round posterior statistics
    """
    results_dir = os.path.abspath(results_dir)
    print(f"Reproducing plots from: {results_dir}")

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = _load_config(results_dir)
    cx_topology     = cfg.get("CX_TOPOLOGY",        "linear").lower()
    sims_per_round  = int(cfg.get("SNPE_SIMS_PER_ROUND", 4000))
    prior_high      = float(cfg.get("PRIOR_HIGH",   0.001))
    print(f"  topology={cx_topology}, sims_per_round={sims_per_round}, "
          f"prior_high={prior_high}")

    # ── Posterior samples ─────────────────────────────────────────────────────
    samples = _load_posterior_samples(results_dir)
    num_params = samples.shape[1]
    print(f"  Posterior samples: {samples.shape[0]} × {num_params}")

    # ── Ground truth (optional) ───────────────────────────────────────────────
    truth = _load_truth(results_dir)
    if truth is not None:
        print(f"  Ground truth loaded ({len(truth)} params).")
    else:
        print("  No ground truth file found; skipping truth overlay.")

    # ── Posterior box plot (all CNOTs) ────────────────────────────────────────
    plot_box(
        samples, num_params, cx_topology, results_dir, truth_np=truth,
        font_scale=font_scale, fig_height=fig_height, fig_width=fig_width,
        marker_scale=marker_scale, truth_marker=truth_marker,
        cnot_label_pad=cnot_label_pad, legend_outside=legend_outside,
        compact_style=compact_box,
    )

    # ── Wrapped multi-row box plot (paper-friendly) ──────────────────────────
    plot_box_wrapped(
        samples, num_params, cx_topology, results_dir, truth_np=truth,
    )

    # ── Scatter inference (all params) ────────────────────────────────────────
    if truth is not None:
        plot_scatter_inference(samples, truth, results_dir, prior_high=prior_high)

    # ── Posterior box plot (user-selected CNOTs) ──────────────────────────────
    if cnot_select is not None:
        n_cnots_total = num_params // 15
        bad = [c for c in cnot_select if c < 1 or c > n_cnots_total]
        if bad:
            raise ValueError(
                f"cnot_select contains out-of-range ctrl-qubit numbers {bad}. "
                f"Valid range: 1–{n_cnots_total} (this model has {n_cnots_total} CNOTs)."
            )
        sel = [c - 1 for c in cnot_select]   # 1-based ctrl-qubit → 0-based index
        tag = "_".join(str(c) for c in cnot_select)
        print(f"  Plotting selected CNOTs {cnot_select} (0-based indices {sel}).")
        plot_box(samples, num_params, cx_topology, results_dir, truth_np=truth,
                 cnot_indices=sel, out_filename=f"paulinoise_box_sel_{tag}.pdf",
                 font_scale=font_scale, fig_height=fig_height, fig_width=fig_width,
                 marker_scale=marker_scale, truth_marker=truth_marker,
                 cnot_label_pad=cnot_label_pad, legend_outside=legend_outside,
                 showfliers=False)

    # ── nMAE scaling plot ─────────────────────────────────────────────────────
    try:
        keep_mask = _gauge_free_keep_mask_from_posterior(samples)
        mae_mean_all, _, round_ids = _compute_nmae(results_dir, prior_high, keep_mask=None)
        mae_mean_gf, _, _ = _compute_nmae(results_dir, prior_high, keep_mask=keep_mask)
        print(f"  Loaded {len(round_ids)} rounds for scaling plot "
              f"(rounds {round_ids[0]}–{round_ids[-1]}).")
        # Prefer the per-round totals saved by training (variable sims per round).
        n_total_path = os.path.join(results_dir, "snpe_n_total_per_round.txt")
        n_total = None
        if os.path.exists(n_total_path):
            per_round_tot = {}
            with open(n_total_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    r_str, n_str, tot_str = line.split(",", 2)
                    per_round_tot[int(r_str)] = int(tot_str)
            n_total = [per_round_tot.get(r, r * sims_per_round) for r in round_ids]
        else:
            n_total = [r * sims_per_round for r in round_ids]

        # Ensure x-axis is truly "cumulative" starting from 0 (round 0 = 0 training data).
        if n_total:
            offset0 = n_total[0]
            if offset0 != 0:
                n_total = [t - offset0 for t in n_total]
        r2_by_round = _load_r2_per_round(results_dir)
        r2_all = [r2_by_round.get(r, (float("nan"), float("nan")))[0] for r in round_ids] if r2_by_round else None
        r2_gf  = [r2_by_round.get(r, (float("nan"), float("nan")))[1] for r in round_ids] if r2_by_round else None
        plot_scaling(
            mae_mean_all,
            mae_mean_gf,
            n_total,
            sims_per_round,
            results_dir,
            r2_all=r2_all,
            r2_gauge_free=r2_gf,
            legend_outside=scaling_legend_outside,
            legend_separate_pdf=scaling_legend_separate_pdf,
            max_n_total=scaling_max_n_total,
            fig_width=scaling_fig_width,
            fig_height=scaling_fig_height,
        )
    except FileNotFoundError as exc:
        print(f"  Scaling plot skipped: {exc}")


# ── R² threshold vs qubits ────────────────────────────────────────────────────

@dataclass(frozen=True)
class Hit:
    results_dir: str
    n_qubits: int
    threshold: float
    n_train: float  # NaN if never reached


def _load_num_params(results_dir: str) -> int:
    path = os.path.join(results_dir, "paulinoise_inference_params.csv")
    with open(path, "r") as f:
        first = f.readline().strip()
    cols = [c for c in first.split(",") if c != ""]
    return len(cols)


def _load_r2_rows(results_dir: str) -> list[tuple[int, int, float, float]]:
    """Return rows of (round, n_train, r2_all, r2_no_gauge)."""
    path = os.path.join(results_dir, "r2_per_round.txt")
    rows: list[tuple[int, int, float, float]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            rows.append((int(parts[0]), int(parts[1]), float(parts[3]), float(parts[4])))
    rows.sort(key=lambda t: (t[1], t[0]))
    return rows


def _first_n_train_at_threshold(rows, *, threshold: float, kind: str) -> float:
    if kind not in ("r2_all", "r2_no_gauge"):
        raise ValueError("kind must be 'r2_all' or 'r2_no_gauge'")
    idx = 2 if kind == "r2_all" else 3
    for row in rows:
        if row[idx] >= threshold:
            return float(row[1])
    return float("nan")


def plot_r2_threshold_vs_qubits_cli(argv: list[str] | None = None) -> int:
    """CLI entry point: training data vs n_qubits at R² thresholds."""
    ap = argparse.ArgumentParser(
        description="Plot training data (n_train) vs n_qubits at R^2 thresholds."
    )
    ap.add_argument("results_dirs", nargs="*")
    ap.add_argument("--more", nargs="+", action="append", default=[], metavar="DIR")
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.99, 0.999])
    ap.add_argument("--r2-kind", choices=["r2_all", "r2_no_gauge"], default="r2_no_gauge")
    ap.add_argument("--out-prefix", default="training_data_vs_qubits")
    args = ap.parse_args(argv)

    more_dirs = [d for group in args.more for d in group]
    results_dirs = list(args.results_dirs) + more_dirs
    if not results_dirs:
        results_dirs = [
            "sbi_paulinoise_results/run_sbi_10q_l3_brickwall_5000data_stop100_6rounds",
            "sbi_paulinoise_results/run_sbi_20q_l3_brickwall_5000data_stop100_6rounds",
            "sbi_paulinoise_results/run_sbi_30q_l3_brickwall_5000data_stop100_6rounds",
            "sbi_paulinoise_results/run_sbi_40q_l3_brickwall_5000data_stop100_6rounds",
            "sbi_paulinoise_results/run_sbi_50q_l3_brickwall_5000data_stop100_5rounds",
        ]

    _rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "mathtext.fontset": "dejavusans",
        "font.size": 11, "axes.labelsize": 12, "legend.fontsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "xtick.direction": "out", "ytick.direction": "out",
        "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    }

    hits: list[Hit] = []
    for d in results_dirs:
        results_dir = os.path.abspath(d)
        num_params = _load_num_params(results_dir)
        n_qubits = _infer_n_qubits(num_params)
        if n_qubits is None:
            raise ValueError(f"Cannot infer n_qubits from num_params={num_params} in {results_dir}")
        rows = _load_r2_rows(results_dir)
        for thr in args.thresholds:
            n_train = _first_n_train_at_threshold(rows, threshold=float(thr), kind=args.r2_kind)
            hits.append(Hit(results_dir=results_dir, n_qubits=int(n_qubits),
                            threshold=float(thr), n_train=n_train))

    qubits = sorted(set(h.n_qubits for h in hits))
    thresholds = list(args.thresholds)
    thr_lo = float(min(thresholds)) if thresholds else 0.99
    thr_hi = float(max(thresholds)) if thresholds else 0.999
    styles_by_thr = {
        thr_lo: dict(fmt="s-", color="#4DB8A8", markerfacecolor="#9DDDD5", markeredgecolor="#2E8A80"),
        thr_hi: dict(fmt="o-", color="#8B65C8", markerfacecolor="#C4A8E8", markeredgecolor="#6040A0"),
    }

    with plt.rc_context(_rc):
        fig, ax = plt.subplots(figsize=(3.7, 2.15))
        for thr in thresholds:
            xs = qubits
            ys = []
            for q in qubits:
                match = next((h for h in hits if h.n_qubits == q and abs(h.threshold - thr) < 1e-15), None)
                ys.append(match.n_train if match is not None else float("nan"))
            st = styles_by_thr.get(thr, dict(fmt="o-", color="#4DB8A8",
                                              markerfacecolor="#9DDDD5", markeredgecolor="#2E8A80"))
            label = (rf"$R^2_{{\text{{gauge-free}}}} \geq {thr}$" if args.r2_kind == "r2_no_gauge"
                     else rf"$R^2_{{\text{{all}}}} \geq {thr}$")
            ax.plot(xs, ys, st["fmt"], color=st["color"], linewidth=1.2, markersize=6,
                    markerfacecolor=st["markerfacecolor"], markeredgecolor=st["markeredgecolor"],
                    markeredgewidth=0.8, label=label)

        ax.set_xlabel("Number of qubits")
        ax.set_ylabel("Training data")
        ax.set_xticks([10, 20, 30, 40, 50])
        ax.set_yticks([10_000, 20_000, 30_000])
        ax.set_yticklabels(["1", "2", "3"])
        ax.set_ylim(8_000, 32_000)
        ax.grid(False)
        ax.annotate(r"$\times10^4$", xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, 4), textcoords="offset points",
                    ha="left", va="bottom", fontsize=10)
        ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.02, 1.06))
        fig.tight_layout()
        fig.savefig(f"{args.out_prefix}.pdf", bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)

    for thr in thresholds:
        print(f"\nThreshold {thr} ({args.r2_kind}):")
        for q in qubits:
            match = next((h for h in hits if h.n_qubits == q and abs(h.threshold - thr) < 1e-15), None)
            val = match.n_train if match is not None else float("nan")
            print(f"  n_qubits={q:>3d}  n_train={val}")

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualize.py <results_dir>")
        sys.exit(1)
    reproduce(sys.argv[1])
