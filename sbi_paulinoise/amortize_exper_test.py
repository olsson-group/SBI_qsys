#!/usr/bin/env python
"""
amortize_exper_test.py — Amortized posterior evaluation for a trained SBI model (round=1).

For each time step t = 1 … n_runs:
  1. Draw θ_t from the chosen drift model.
  2. Generate an observation x via a selectable simulator backend
     (default: paulinoise_simulator_1.NoisyCircuitSimulator / aer_dm).
  3. Draw posterior samples from the loaded estimator conditioned on x.
  4. Compute per-parameter MAE between posterior mean and θ_t.

Drift model (--drift_mode sinusoidal): θ_t oscillates around a fixed θ₀ with amplitude
drift_scale·prior_high per parameter and a random phase; the period is set by --drift_period
(default: n_runs). Also saves drift_trajectory_IX_XX_ZZ.pdf showing the IX, XX and ZZ
trajectories of CNOT1→2 as a visual sanity check.

θ_t is clipped to [0, prior_high] to keep parameters physical.
Reproducibility is controlled by --drift_seed (it sets θ₀ and the drift trajectory).

Outputs (saved in sbi_paulinoise_results/amortize_exper/<results_dir_name>/):
  - amortize_results.csv          — run, θ_true, observation, posterior mean/std, MAE
  - metadata.json                 — full config for reproduce()
  - posterior_samples_run{i}.npy  — one file per time step
  - paulinoise_box_run{i}.pdf            — one per time step

Usage:
    python sbi_paulinoise/amortize_exper_test.py --n_runs 5
    python sbi_paulinoise/amortize_exper_test.py --n_runs 10 --drift_scale 0.4 --drift_period 5 --drift_seed 42
"""

import argparse
import csv
import json
import os
import sys
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Project root is the parent of this file's directory (sbi_paulinoise/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sbi_paulinoise.visualize import _VIZ_RC, _GATE_COLORS, _make_boxplot, _style_ax


def _get_noisy_simulator_class(backend: str):
    """Return NoisyCircuitSimulator class for the requested backend."""
    backend = (backend or "sim1").strip().lower()
    if backend in {"sim2", "paulinoise_simulator_2", "stim"}:
        from sbi_paulinoise.paulinoise_simulator_2 import NoisyCircuitSimulator as _Sim
        return _Sim, "sim2"
    if backend in {"sim1", "paulinoise_simulator_1", "aer", "aer_dm"}:
        from sbi_paulinoise.paulinoise_simulator_1 import NoisyCircuitSimulator as _Sim
        return _Sim, "sim1"
    raise ValueError(f"Unknown simulator backend '{backend}'. Use 'sim1' or 'sim2'.")


# ─────────────────────────────────────────────────────────────────────────────
# Config parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_config(config_path: str) -> dict:
    """Parse config_snapshot.txt into a typed dict."""
    bool_keys  = {"SNPE_REFRESH_MODEL"}
    int_keys   = {"N_QUBITS", "EPOCHS", "BATCH_SIZE", "STOP_AFTER_EPOCHS",
                  "NUM_POSTERIOR_SAMPLES", "NUM_TRANSFORMS", "NSF_NUM_BINS",
                  "ZUKO_GF_COMPONENTS", "NPE_N_SIMS", "NPE_SIM_N_JOBS",
                  "SNPE_NUM_ROUNDS", "SNPE_SIMS_PER_ROUND", "SNPE_SIM_N_JOBS"}
    float_keys = {"PRIOR_HIGH", "LR", "CLIP_MAX_NORM", "WEIGHT_DECAY"}
    cfg = {}
    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if ": " not in line or line.startswith("SBI run"):
                continue
            key, val = line.split(": ", 1)
            key, val = key.strip(), val.strip()
            if key in bool_keys:
                cfg[key] = val.lower() == "true"
            elif key in int_keys:
                try:
                    cfg[key] = int(val)
                except ValueError:
                    cfg[key] = val
            elif key in float_keys:
                try:
                    cfg[key] = float(val)
                except ValueError:
                    cfg[key] = val
            else:
                cfg[key] = val
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Posterior loading
# ─────────────────────────────────────────────────────────────────────────────

def load_posterior(results_dir: str, config: dict, num_params: int, device: str, infer_high: float | None = None):
    """Load estimator.pt and wrap it in an sbi posterior for amortized sampling."""
    from sbi.inference import NPE
    from sbi.utils import BoxUniform

    prior_high = float(config["PRIOR_HIGH"])
    bound = infer_high if infer_high is not None else prior_high
    prior = BoxUniform(
        low=torch.zeros(num_params, device=device),
        high=torch.ones(num_params, device=device) * bound,
    )
    estimator_path = os.path.join(results_dir, "estimator.pt")
    print(f"  Loading estimator from {estimator_path} ...")
    estimator = torch.load(estimator_path, map_location=device, weights_only=False)
    estimator.eval()

    inference = NPE(prior=prior, device=device)
    posterior = inference.build_posterior(estimator)

    return posterior, prior


# ─────────────────────────────────────────────────────────────────────────────
# Posterior sampling with non-negativity rejection
# ─────────────────────────────────────────────────────────────────────────────

def sample_posterior_for_obs(
    posterior,
    x_obs_t: torch.Tensor,
    num_samples: int,
    prior_high: float,
    max_iter: int = 8,
    batch_size: int = None,  # ignored; kept for API compatibility
    *,
    verbose: bool = False,
) -> np.ndarray:
    """Draw `num_samples` in-support [0, prior_high] posterior samples.

    Samples all at once (single posterior.sample call in the typical case) to
    avoid repeated normalizing-flow overhead from many small calls.
    Falls back to doubling the draw size if rejection is heavy.
    """
    # Keep sbi's BoxUniform prior rejection: it confines samples to [0, prior_high] per
    # dimension. Without it the flow's full R^d support is exposed, and in d=700+ the
    # joint probability of every coordinate landing in [0, prior_high] is essentially zero.
    collected, n_collected = [], 0
    draw = num_samples  # request everything in one shot
    total_drawn = 0
    total_valid = 0
    with torch.inference_mode():
        for it in range(max_iter):
            if n_collected >= num_samples:
                break
            batch = posterior.sample((draw,), x=x_obs_t, show_progress_bars=False)
            # Do rejection on-device to avoid copying rejected samples to CPU.
            mask = (batch >= 0).all(dim=1) & (batch <= prior_high).all(dim=1)
            if torch.any(mask):
                valid_t = batch[mask]
                valid = valid_t.detach().cpu().numpy()
                collected.append(valid)
                n_collected += len(valid)
                n_valid = int(len(valid))
            else:
                n_valid = 0
            total_drawn += int(draw)
            total_valid += int(n_valid)
            accept_rate = n_valid / max(int(draw), 1)
            remaining = num_samples - n_collected
            if verbose:
                dev = str(x_obs_t.device)
                print(
                    f"    [posterior.sample] it={it+1}/{max_iter} device={dev} "
                    f"draw={int(draw)} valid={n_valid} accept_rate={accept_rate:.3f} "
                    f"collected={n_collected}/{num_samples}"
                )
            if remaining <= 0:
                break
            # If most samples were rejected, increase draw to recover faster.
            draw = min(int(remaining / max(accept_rate, 0.01)) + 100, max(num_samples * 8, 512))
    if not collected:
        raise RuntimeError("No valid posterior samples obtained (all outside [0, prior_high]).")
    samples = np.concatenate(collected, axis=0)[:num_samples]
    if len(samples) < num_samples:
        print(f"  Warning: only {len(samples)}/{num_samples} valid samples obtained.")
    # Attach lightweight diagnostics for caller (no heavy tensors).
    sample_posterior_for_obs.last_stats = {
        "total_drawn": int(total_drawn),
        "total_valid": int(total_valid),
        "overall_accept_rate": float(total_valid / max(total_drawn, 1)),
        "n_iters": int(min(max_iter, len(collected) if collected else 0)),
    }
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.frameon": False,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}

# RC for drift-trajectory plot — matches training_data_vs_qubits.pdf style
_DRIFT_RC = {
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "DejaVu Sans", "Helvetica"],
    "mathtext.fontset":   "dejavusans",
    "font.size":          13,
    "axes.labelsize":     14,
    "legend.fontsize":    11,
    "xtick.labelsize":    12,
    "ytick.labelsize":    12,
    "axes.linewidth":     0.7,
    "xtick.major.width":  0.7,
    "ytick.major.width":  0.7,
    "xtick.major.size":   3.0,
    "ytick.major.size":   3.0,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "legend.frameon":     False,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
}


def plot_paulinoise_box(
    posterior_samples_np: np.ndarray,
    num_params: int,
    cx_topology: str,
    truth_np: np.ndarray,
    out_path: str,
    run_label: str = "",
):
    """paulinoise_box-style posterior boxplot for one run (replicates visualize.plot_box)."""
    if num_params % 15 != 0:
        raise ValueError(f"Cannot infer param structure from num_params={num_params}")
    n_cnots = num_params // 15
    n_gate_params = num_params

    bases = ["I", "X", "Y", "Z"]
    pauli_2q_raw = [p1 + p2 for p1 in bases for p2 in bases
                    if not (p1 == "I" and p2 == "I")]
    w1 = [p for p in pauli_2q_raw if "I" in p]
    w2 = [p for p in pauli_2q_raw if "I" not in p]
    pauli_2q  = w1 + w2
    old_pos   = {lbl: idx for idx, lbl in enumerate(pauli_2q_raw)}
    perm_in_g = [old_pos[lbl] for lbl in pauli_2q]
    col_perm  = [i * 15 + j for i in range(n_cnots) for j in perm_in_g]

    cnot_pairs  = [(i, i + 1) for i in range(n_cnots)]
    gate_labels = [f"$\\mathrm{{{lbl}}}$" for _ in range(n_cnots) for lbl in pauli_2q]
    gate_data   = posterior_samples_np[:, col_perm]

    with plt.rc_context(_VIZ_RC):
        fw = max(10.0, n_gate_params * 0.32)
        fig, ax_g = plt.subplots(figsize=(fw, 4))

        for i, (ctl, tgt) in enumerate(cnot_pairs):
            lo, hi = i * 15 - 0.5, i * 15 + 14.5
            ax_g.axvspan(lo, hi, color=_GATE_COLORS[i % len(_GATE_COLORS)], alpha=0.45, zorder=0)
            ax_g.text(
                (lo + hi) / 2, -0.20,
                f"$\\mathrm{{CNOT}}_{{{ctl+1}\\rightarrow{tgt+1}}}$",
                ha="center", va="top",
                transform=ax_g.get_xaxis_transform(),
                fontsize=14, color="black",
            )

        _make_boxplot(ax_g, gate_data, n_gate_params)

        legend_handles = [
            mpatches.Patch(facecolor="#9DDDD5", edgecolor="black",
                           linewidth=0.8, label="Posterior samples"),
        ]
        gate_truth = truth_np[col_perm]
        ax_g.plot(range(n_gate_params), gate_truth,
                  "x", color="#E8000D", markersize=6, markeredgewidth=1.2, zorder=10)
        legend_handles.append(
            Line2D([], [], marker="x", color="#E8000D", linestyle="none",
                   markersize=6, markeredgewidth=1.2, label="Ground truth")
        )

        leg = ax_g.legend(handles=legend_handles, loc="upper right",
                          frameon=True, facecolor="white", framealpha=0.85, edgecolor="none")
        leg.set_zorder(20)
        fmt_g = _style_ax(ax_g, n_gate_params)
        ax_g.set_ylabel("Posterior value")

        ax_g.set_xticks(range(n_gate_params))
        split_labels_g = [lbl.replace("$\\mathrm{", "").replace("}$", "")
                          for lbl in gate_labels]
        split_labels_g = [f"$\\mathrm{{{lbl[0]}}}$\n$\\mathrm{{{lbl[1]}}}$"
                          for lbl in split_labels_g]
        ax_g.set_xticklabels(split_labels_g, rotation=0, ha="center")
        ax_g.tick_params(axis="both", direction="out")
        fig.canvas.draw()
        off = fmt_g.get_offset()
        if off:
            ax_g.yaxis.offsetText.set_visible(True)
            ax_g.yaxis.offsetText.set_text(off)

        fig.tight_layout()
        fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved posterior box plot → {out_path}")


def plot_drift_trajectory(
    theta_sequence: list,
    post_means: list,
    post_stds: list,
    out_path: str,
    prior_high: float,
    num_params: int = None,
    third_pauli: str = "ZZ",
):
    """Plot drift trajectory of the first CNOT's IX/XX/(third): truth and posterior mean ± std.

    Solid lines = ground truth θ; dashed lines + shaded band = posterior mean ± std.
    The CNOT index is fixed to the first entangling gate (index 0).
    """

    first = 0                            # first CNOT index
    # pauli_2q_raw ordering (15 terms):
    #   IX=0, IY=1, IZ=2, XI=3, XX=4, XY=5, XZ=6,
    #   YI=7, YX=8, YY=9, YZ=10, ZI=11, ZX=12, ZY=13, ZZ=14
    _pauli_2q_order = ["IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ"]
    third_pauli = str(third_pauli).upper()
    if third_pauli not in _pauli_2q_order:
        raise ValueError(f"third_pauli must be one of {_pauli_2q_order}, got {third_pauli!r}")

    idx_ix = first * 15 + _pauli_2q_order.index("IX")
    idx_xx = first * 15 + _pauli_2q_order.index("XX")
    idx_third = first * 15 + _pauli_2q_order.index(third_pauli)

    t_steps = np.arange(len(theta_sequence))

    truth_ix    = np.array([th[idx_ix]    for th in theta_sequence])
    truth_xx    = np.array([th[idx_xx]    for th in theta_sequence])
    truth_third = np.array([th[idx_third] for th in theta_sequence])
    mean_ix     = np.array([m[idx_ix]     for m in post_means])
    mean_xx     = np.array([m[idx_xx]     for m in post_means])
    mean_third  = np.array([m[idx_third]  for m in post_means])
    std_ix      = np.array([s[idx_ix]     for s in post_stds])
    std_xx      = np.array([s[idx_xx]     for s in post_stds])
    std_third   = np.array([s[idx_third]  for s in post_stds])

    col = "#4DB8A8"

    # Shared scale across all panels
    vmax = max(np.max(np.abs(truth_ix)), np.max(np.abs(truth_xx)), np.max(np.abs(truth_third)),
               np.max(np.abs(mean_ix + std_ix)), np.max(np.abs(mean_xx + std_xx)),
               np.max(np.abs(mean_third + std_third)))
    exp   = int(np.floor(np.log10(vmax))) if vmax > 0 else 0
    scale = 10 ** exp

    def _style_panel(ax, vals_truth, vals_mean, vals_std, pauli_sub):
        ax.plot(t_steps, vals_truth / scale, linestyle="-", linewidth=1.5,
                color="#8B65C8", label="Truth")
        ax.plot(t_steps, vals_mean / scale, linestyle="-", linewidth=1.2,
                marker="o", markersize=6, color=col,
                markerfacecolor="#9DDDD5", markeredgecolor="#2E8A80", markeredgewidth=0.8, label="Posterior mean")
        ax.fill_between(t_steps,
                        (vals_mean - vals_std) / scale,
                        (vals_mean + vals_std) / scale,
                        color=col, alpha=0.2, label=r"$\pm$s.d.")
        ax.set_ylabel(rf"$p_{{\mathrm{{{pauli_sub}}}}}$", fontsize=14)
        ax.set_xlim(-0.3, len(theta_sequence) - 0.7)
        ax.set_xticks([0, 3, 6, 9])
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=2, prune=None))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{x:.1f}".rstrip('0').rstrip('.')
        ))
        ax.tick_params(axis="both", direction="out")

    with plt.rc_context(_DRIFT_RC):
        fig, (ax_ix, ax_xx, ax_third) = plt.subplots(3, 1, figsize=(3.7, 4.0),
                                                      sharex=True,
                                                      gridspec_kw={"hspace": 0.08})

        _style_panel(ax_ix,    truth_ix,    mean_ix,    std_ix,    "IX")
        _style_panel(ax_xx,    truth_xx,    mean_xx,    std_xx,    "XX")
        _style_panel(ax_third, truth_third, mean_third, std_third, third_pauli)

        # Enforce exactly 3 ticks per panel
        fig.canvas.draw()
        fmt = ticker.FuncFormatter(lambda x, _: f"{x:.1f}".rstrip('0').rstrip('.'))
        for ax in (ax_ix, ax_xx, ax_third):
            ylim = ax.get_ylim()
            ticks = [t for t in ax.get_yticks() if ylim[0] <= t <= ylim[1]]
            if len(ticks) > 3:
                idx = np.round(np.linspace(0, len(ticks) - 1, 3)).astype(int)
                ticks = [ticks[i] for i in idx]
            elif len(ticks) < 3:
                ticks = np.linspace(ylim[0], ylim[1], 3).tolist()
            ax.set_yticks(ticks)
            span = ticks[-1] - ticks[0]
            ax.set_ylim(ticks[0] - span * 0.2, ticks[-1] + span * 0.2)
            ax.yaxis.set_major_formatter(fmt)

        ax_xx.legend(loc="lower right", fontsize=9, bbox_to_anchor=(1.03, -0.02),
                     handlelength=1.2, handletextpad=0.4, labelspacing=0.25, markerscale=0.8,
                     borderpad=0.4)
        ax_ix.tick_params(labelbottom=False)
        ax_xx.tick_params(labelbottom=False)
        ax_third.set_xlabel("Run index")
        # Shared scale annotation at top-left of upper panel
        ax_ix.annotate(rf"$\times10^{{{exp}}}$", xy=(0, 1), xycoords="axes fraction",
                       xytext=(0, 4), textcoords="offset points",
                       ha="left", va="bottom", fontsize=12)
        fig.subplots_adjust(left=0.12, right=0.97, top=0.95, bottom=0.12)
        fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved drift trajectory plot → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Time-drift parameter sequence generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_drifting_thetas(
    rng: np.random.Generator,
    num_params: int,
    prior_high: float,
    n_steps: int,
    drift_mode: str = "sinusoidal",
    drift_scale: float = 0.3,
    drift_period: float = None,
    theta_0_fixed: np.ndarray | None = None,
) -> list:
    """Generate a sequence of θ vectors over n_steps time steps.

    Parameters
    ----------
    rng           : seeded numpy Generator (controls θ₀ and all stochastic components)
    num_params    : dimensionality of θ
    prior_high    : prior upper bound; used for scaling amplitudes and clipping output
    n_steps       : number of time steps
    drift_mode    : "sinusoidal"
    drift_scale   : amplitude of drift relative to prior_high
    drift_period  : sinusoidal period in steps (default: n_steps)
    theta_0_fixed : if provided, use as the starting point θ₀ instead of random sampling.
                    Oscillation center is clamped to [drift_scale, 1-drift_scale]*prior_high
                    so params above 0.7 drift downward, below 0.3 drift upward.

    Returns
    -------
    list of n_steps arrays, each shape (num_params,), all values in [0, prior_high]
    """
    if drift_mode == "sinusoidal":
        if drift_scale >= 0.5:
            raise ValueError(f"drift_scale={drift_scale} must be < 0.5 for sinusoidal mode "
                             "to guarantee θ₀ stays within [0, prior_high].")
        period = drift_period if drift_period is not None else float(n_steps)
        amplitudes = np.full(num_params, drift_scale * prior_high)

        if theta_0_fixed is not None:
            # Clamp oscillation center based on actual amplitude so center±amp ∈ [0, prior_high].
            # This guarantees theta(t=0) = theta_0_fixed exactly for all parameters.
            theta_center = np.clip(theta_0_fixed, amplitudes, prior_high - amplitudes)
            # Set phase so theta(t=0) = theta_0_fixed exactly
            # theta(0) = theta_center + A*sin(phi) = theta_0_fixed
            with np.errstate(invalid="ignore"):
                sin_phi = np.where(
                    amplitudes > 0,
                    np.clip((theta_0_fixed - theta_center) / amplitudes, -1.0, 1.0),
                    0.0,
                )
            base_phi = np.arcsin(sin_phi)
            # For params starting above center: use π-arcsin branch so slope is negative
            # (drift downward initially). For params at/below center: use arcsin branch
            # (slope positive, drift upward initially).
            phases = np.where(theta_0_fixed > theta_center, np.pi - base_phi, base_phi)
        else:
            # Original behaviour: random θ₀ and random phase
            theta_center = rng.uniform(drift_scale * prior_high,
                                       (1.0 - drift_scale) * prior_high,
                                       size=num_params)
            phases = rng.uniform(0.0, 2 * np.pi, size=num_params)

        thetas = []
        for t in range(n_steps):
            theta_t = theta_center + amplitudes * np.sin(2 * np.pi * t / period + phases)
            thetas.append(np.clip(theta_t, 0.0, prior_high))
        return thetas

    raise ValueError(f"Unknown drift_mode={drift_mode!r}. Only 'sinusoidal' is supported.")


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    results_dir: str,
    num_posterior_samples: int = 2000,
    n_runs: int = 5,
    out_dir: str = "sbi_paulinoise_results/amortize_exper",
    drift_mode: str = "sinusoidal",
    drift_scale: float = 0.3,
    drift_period: float = None,
    drift_seed: int = 0,
    theta_0_file: str | None = None,
    simulator_backend: str = "sim1",
    infer_upper_factor: float | None = None,
    drift_third_pauli: str = "ZI",
):
    """Run amortized posterior evaluation over n_runs time steps.

    Parameters
    ----------
    results_dir           : path to trained SBI results folder (contains estimator.pt,
                            config_snapshot.txt, and observables CSV)
    num_posterior_samples : posterior samples to draw per time step
    n_runs                : number of time steps to simulate
    out_dir               : output base directory (sbi_paulinoise_results/amortize_exper/<results_dir_name>/ is created)
    drift_mode            : "sinusoidal" — how θ evolves over time
    drift_scale           : drift amplitude relative to prior_high
    drift_period          : sinusoidal period in steps (default: n_runs)
    drift_seed            : RNG seed controlling θ₀ and the full drift trajectory
    theta_0_file          : path to CSV of ground-truth θ₀ (e.g. ground_truth_8q/linear/paulinoise_1_parameters.csv).
                            If provided, used as fixed starting point; params > 0.7*prior_high
                            drift downward, params < 0.3*prior_high drift upward.
    """
    # Resolve results_dir relative to project root if not absolute
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(ROOT, results_dir)

    # Create subdirectory named after the results folder
    run_name = os.path.basename(os.path.normpath(results_dir))
    out_dir = os.path.join(ROOT, out_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # ── Parse config ─────────────────────────────────────────────────────────
    config_path = os.path.join(results_dir, "config_snapshot.txt")
    config = parse_config(config_path)
    n_qubits      = config["N_QUBITS"]
    cx_topology   = config["CX_TOPOLOGY"]
    prior_high    = float(config["PRIOR_HIGH"])
    _factor = infer_upper_factor if infer_upper_factor is not None else float(config.get("INFER_UPPER_FACTOR", 1.0))
    infer_high    = prior_high * _factor
    learning_mode = config.get("LEARNING_MODE", "obs_learning")
    print(f"Config: N_QUBITS={n_qubits}, CX_TOPOLOGY={cx_topology}, "
          f"PRIOR_HIGH={prior_high}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    SimClass, sim_kind = _get_noisy_simulator_class(simulator_backend)
    print(f"Simulator backend: {sim_kind}")

    # ── Load observables ──────────────────────────────────────────────────────
    obs_file = config.get("OBSERVABLES_FILE", "")
    if not os.path.isabs(obs_file):
        obs_file = os.path.join(ROOT, obs_file)
    observables = []
    with open(obs_file) as f:
        for row in csv.reader(f):
            if row and row[0] != "Observable String":
                observables.append(row[0])
    n_obs = len(observables)
    print(f"Observables: {n_obs} strings from {obs_file}")

    # ── Determine num_params ──────────────────────────────────────────────────
    _tmp = SimClass(n_qubits=n_qubits, cx_topology=cx_topology)
    num_params = _tmp.total_params
    del _tmp
    print(f"num_params={num_params}")

    # ── Load posterior ────────────────────────────────────────────────────────
    posterior, prior = load_posterior(results_dir, config, num_params, device, infer_high=infer_high)
    print("Posterior loaded.")

    # ── Derive standardization stats from round-1 training data (if STANDARDIZE_X=True) ──
    x_std_mean: np.ndarray | None = None
    x_std_std:  np.ndarray | None = None
    if config.get("STANDARDIZE_X", "").lower() in {"true", "1", "yes"}:
        _x_train_path = os.path.join(results_dir, "x_train_round1.npy")
        if os.path.exists(_x_train_path):
            _x_train = np.load(_x_train_path)
            x_std_mean = _x_train.mean(axis=0).astype(np.float32)
            x_std_std  = _x_train.std(axis=0).clip(1e-4).astype(np.float32)
            del _x_train
            print(f"Standardizing obs from x_train_round1: "
                  f"mean∈[{x_std_mean.min():.4f},{x_std_mean.max():.4f}], "
                  f"std∈[{x_std_std.min():.2e},{x_std_std.max():.2e}]")
        else:
            print(f"Warning: STANDARDIZE_X=True but {_x_train_path} not found — "
                  "passing raw observations to posterior.")

    # ── Generate drifting θ sequence ──────────────────────────────────────
    theta_0_fixed = None
    if theta_0_file is not None:
        _path = theta_0_file if os.path.isabs(theta_0_file) else os.path.join(ROOT, theta_0_file)
        theta_0_fixed = np.loadtxt(_path, delimiter=",").flatten()[:num_params]
        print(f"θ₀ loaded from {_path}")
    drift_rng = np.random.default_rng(drift_seed)
    torch.manual_seed(drift_seed)
    theta_sequence = generate_drifting_thetas(
        rng=drift_rng,
        num_params=num_params,
        prior_high=prior_high,
        n_steps=n_runs,
        drift_mode=drift_mode,
        drift_scale=drift_scale,
        drift_period=drift_period,
        theta_0_fixed=theta_0_fixed,
    )
    print(f"Drift mode: {drift_mode}  (scale={drift_scale}, n_runs={n_runs}, drift_seed={drift_seed})")

    all_rows:               list[dict]       = []
    all_post_samples:       list[np.ndarray] = []
    all_theta_true:         list[np.ndarray] = []
    all_post_means:         list[np.ndarray] = []
    all_post_stds:          list[np.ndarray] = []
    sampling_timing_rows:   list[dict]       = []  # per-instance posterior sampling timings

    # Pre-create sim2 once outside the loop (no per-run state; stim circuit build is expensive)
    if sim_kind == "sim2":
        sim_shared = SimClass(n_qubits=n_qubits, cx_topology=cx_topology)

    # ── Experiment loop ───────────────────────────────────────────────────────
    for run_idx in range(n_runs):
        sim_seed = drift_seed * 10000 + run_idx   # deterministic, no separate seeds needed
        print(f"\n── Time step {run_idx + 1}/{n_runs} ──")

        # float64 — Qiskit AerJSONEncoder cannot serialize numpy float32
        theta_true = theta_sequence[run_idx]

        if sim_kind == "sim1":
            sim = SimClass(
                n_qubits=n_qubits,
                cx_topology=cx_topology,
                random_seed=sim_seed,
            )
        else:
            sim = sim_shared
        print(f"  Generating observation ({sim_kind}) ...")
        if learning_mode == "shots_learning":
            if sim_kind == "sim1":
                obs_arr = sim.compute_bitstring_probs(theta_true, method="aer_dm")
            else:
                obs_arr = sim.compute_bitstring_probs(theta_true)
        else:
            if sim_kind == "sim1":
                obs_list = sim.compute_expectations(theta_true, observables, method="aer_dm")
            else:
                obs_list = sim.compute_expectations(theta_true, observables)
            obs_arr  = np.array(obs_list, dtype=np.float32)
        print(f"  Observation shape: {obs_arr.shape}")

        if x_std_mean is not None:
            obs_arr_in = np.clip((obs_arr - x_std_mean) / x_std_std, -10.0, 10.0).astype(np.float32)
        else:
            obs_arr_in = obs_arr
        x_obs_t = torch.as_tensor(obs_arr_in, dtype=torch.float32).unsqueeze(0).to(device)
        print(f"  Sampling {num_posterior_samples} posterior samples ...")
        _t0 = time.perf_counter()
        samples = sample_posterior_for_obs(
            posterior,
            x_obs_t,
            num_posterior_samples,
            infer_high,
            verbose=True,
        )
        _dt = time.perf_counter() - _t0
        _st = getattr(sample_posterior_for_obs, "last_stats", {}) or {}
        sampling_timing_rows.append(
            {
                "run": int(run_idx + 1),
                "n_samples": int(num_posterior_samples),
                "total_seconds": float(_dt),
                "avg_ms_per_sample": float((_dt / max(1, int(num_posterior_samples))) * 1e3),
                "total_drawn": int(_st.get("total_drawn", 0)),
                "overall_accept_rate": float(_st.get("overall_accept_rate", float("nan"))),
            }
        )
        print(
            f"  Posterior sampling time: {_dt:.4f}s  "
            f"({sampling_timing_rows[-1]['avg_ms_per_sample']:.3f} ms/sample)"
        )

        post_mean   = samples.mean(axis=0)
        post_median = np.median(samples, axis=0)
        post_std    = samples.std(axis=0)
        mae_per_param        = np.abs(post_mean   - theta_true)
        mae_median_per_param = np.abs(post_median - theta_true)
        nmae_mean   = float(mae_per_param.mean()        / prior_high * 100)
        nmae_median = float(mae_median_per_param.mean() / prior_high * 100)

        all_post_samples.append(samples)
        all_theta_true.append(theta_true)
        all_post_means.append(post_mean)
        all_post_stds.append(post_std)
        print(f"  nMAE (mean): {nmae_mean:.2f}%   nMAE (median): {nmae_median:.2f}%")

        np.save(
            os.path.join(out_dir, f"posterior_samples_run{run_idx+1}.npy"),
            samples,
        )

        row: dict = {
            "run": run_idx,
            "nmae_mean_pct":   round(nmae_mean, 4),
            "nmae_median_pct": round(nmae_median, 4),
        }
        for i in range(num_params):
            row[f"theta_true_{i}"] = float(theta_true[i])
        for i in range(len(obs_arr)):
            row[f"obs_{i}"] = float(obs_arr[i])
        for i in range(num_params):
            row[f"post_mean_{i}"] = float(post_mean[i])
        for i in range(num_params):
            row[f"post_std_{i}"]  = float(post_std[i])
        for i in range(num_params):
            row[f"mae_mean_{i}"]   = float(mae_per_param[i])
            row[f"mae_median_{i}"] = float(mae_median_per_param[i])
        all_rows.append(row)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "amortize_results.csv")
    if all_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
    print(f"\nSaved results CSV → {csv_path}  ({len(all_rows)} rows)")

    # ── Save posterior sampling timing summary ────────────────────────────────
    timing_path = os.path.join(out_dir, "posterior_sampling_timing.txt")
    if sampling_timing_rows:
        totals = np.array([r["total_seconds"] for r in sampling_timing_rows], dtype=float)
        avgs_ms = np.array([r["avg_ms_per_sample"] for r in sampling_timing_rows], dtype=float)
        with open(timing_path, "w") as f:
            f.write("# Posterior sampling timing per instance\n")
            f.write("# columns: instance,n_samples,total_seconds,avg_ms_per_sample,total_drawn,overall_accept_rate\n")
            for r in sampling_timing_rows:
                f.write(
                    f"{r['run']},{r['n_samples']},{r['total_seconds']:.6f},{r['avg_ms_per_sample']:.6f},"
                    f"{int(r.get('total_drawn', 0))},{float(r.get('overall_accept_rate', float('nan'))):.6f}\n"
                )
            f.write("# summary\n")
            f.write(f"# total_seconds: mean={totals.mean():.6f}, std={totals.std(ddof=0):.6f}\n")
            f.write(f"# avg_ms_per_sample: mean={avgs_ms.mean():.6f}, std={avgs_ms.std(ddof=0):.6f}\n")
        print(f"Saved posterior sampling timing → {timing_path}")

    # ── Save metadata ─────────────────────────────────────────────────────────
    metadata = {
        "cx_topology":  cx_topology,
        "prior_high":   prior_high,
        "num_params":   num_params,
        "n_runs":       n_runs,
        "drift_mode":   drift_mode,
        "drift_scale":  drift_scale,
        "drift_period": drift_period,
        "drift_seed":   drift_seed,
        "num_posterior_samples": int(num_posterior_samples),
        "drift_third_pauli": str(drift_third_pauli).upper(),
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # ── Plots ─────────────────────────────────────────────────────────────────
    for run_idx in range(n_runs):
        plot_paulinoise_box(
            posterior_samples_np=all_post_samples[run_idx],
            num_params=num_params,
            cx_topology=cx_topology,
            truth_np=all_theta_true[run_idx],
            out_path=os.path.join(out_dir, f"paulinoise_box_run{run_idx+1}.pdf"),
            run_label=f"Time step {run_idx+1}",
        )

    if drift_mode == "sinusoidal":
        _third = str(drift_third_pauli).upper()
        plot_drift_trajectory(
            theta_sequence=theta_sequence,
            post_means=all_post_means,
            post_stds=all_post_stds,
            out_path=os.path.join(out_dir, f"drift_trajectory_IX_XX_{_third}.pdf"),
            prior_high=prior_high,
            num_params=num_params,
            third_pauli=_third,
        )

    print(f"\nDone. All outputs in {out_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# reproduce — regenerate all plots from a saved amortize_exper sub-directory
# ─────────────────────────────────────────────────────────────────────────────

def reproduce(out_dir: str) -> None:
    """Regenerate all plots from a saved amortize_exper output directory.

    Reads:
        metadata.json
        amortize_results.csv
        posterior_samples_run{i}.npy

    Saves (overwrites):
        paulinoise_box_run{i}.pdf
        drift_trajectory_IX_XX_{third}.pdf  (third is from metadata.json if present, else ZI)
    """
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(ROOT, out_dir)
    out_dir = os.path.normpath(out_dir)
    print(f"Reproducing plots from: {out_dir}")

    with open(os.path.join(out_dir, "metadata.json")) as f:
        meta = json.load(f)
    cx_topology = meta["cx_topology"]
    prior_high  = float(meta["prior_high"])
    num_params  = int(meta["num_params"])
    n_runs      = int(meta["n_runs"])
    drift_mode  = meta.get("drift_mode", "sinusoidal")
    drift_third_pauli = str(meta.get("drift_third_pauli", "ZI")).upper()
    print(f"  cx_topology={cx_topology}, prior_high={prior_high}, "
          f"num_params={num_params}, n_runs={n_runs}, drift_mode={drift_mode}")

    csv_path = os.path.join(out_dir, "amortize_results.csv")
    all_theta_true:           list[np.ndarray] = []
    all_post_means: list[np.ndarray] = []
    all_post_stds:  list[np.ndarray] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            theta  = np.array([float(row[f"theta_true_{i}"]) for i in range(num_params)])
            pmean  = np.array([float(row[f"post_mean_{i}"])  for i in range(num_params)])
            pstd   = np.array([float(row[f"post_std_{i}"])   for i in range(num_params)])
            all_theta_true.append(theta)
            all_post_means.append(pmean)
            all_post_stds.append(pstd)

    for run_idx in range(n_runs):
        npy_path = os.path.join(out_dir, f"posterior_samples_run{run_idx+1}.npy")
        if not os.path.exists(npy_path):
            print(f"  Warning: {npy_path} not found — skipping paulinoise_box for run {run_idx+1}")
            continue
        samples = np.load(npy_path)
        plot_paulinoise_box(
            posterior_samples_np=samples,
            num_params=num_params,
            cx_topology=cx_topology,
            truth_np=all_theta_true[run_idx],
            out_path=os.path.join(out_dir, f"paulinoise_box_run{run_idx+1}.pdf"),
            run_label=f"Time step {run_idx+1}",
        )

    if drift_mode == "sinusoidal":
        plot_drift_trajectory(
            theta_sequence=all_theta_true,
            post_means=all_post_means,
            post_stds=all_post_stds,
            out_path=os.path.join(out_dir, f"drift_trajectory_IX_XX_{drift_third_pauli}.pdf"),
            prior_high=prior_high,
            num_params=num_params,
            third_pauli=drift_third_pauli,
        )

    print(f"All plots saved to {out_dir}/")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Amortized posterior evaluation for a trained round-1 SBI model."
    )
    parser.add_argument(
        "--results_dir",
        default="sbi_paulinoise_results/run_sbi_50q_l3_brickwall_30000data_stop100_1rounds",
        help="SBI results folder (relative to project root, or absolute)",
    )
    parser.add_argument(
        "--n_runs", type=int, default=10,
        help="Number of time steps to simulate (default: 10)",
    )
    parser.add_argument(
        "--num_posterior_samples", type=int, default=2000,
        help="Posterior samples to draw per run",
    )
    parser.add_argument(
        "--out_dir", default="sbi_paulinoise_results/amortize_exper",
        help="Output base directory (relative to project root, or absolute)",
    )
    parser.add_argument(
        "--drift_mode", default="sinusoidal",
        choices=["sinusoidal"],
        help="Time-drift model for θ (only sinusoidal is available)",
    )
    parser.add_argument(
        "--drift_scale", type=float, default=0.15,
        help="Drift amplitude relative to prior_high (default: 0.3)",
    )
    parser.add_argument(
        "--drift_period", type=float, default=None,
        help="Period in steps for sinusoidal drift (default: n_runs)",
    )
    parser.add_argument(
        "--drift_seed", type=int, default=42,
        help="RNG seed controlling both initial θ₀ and drift trajectory (default: 0)",
    )
    parser.add_argument(
        "--theta_0_file", type=str, default=None,
    help="CSV file of ground-truth θ₀ to use as fixed starting point (e.g. ground_truth_8q/linear/paulinoise_1_parameters.csv)",
    )
    parser.add_argument(
        "--simulator_backend",
        default="sim2",
        choices=["sim1", "sim2"],
        help="Forward-model backend for generating observations: sim1=Qiskit Aer density_matrix (default), sim2=Stim/Heisenberg (memory-light).",
    )
    parser.add_argument(
        "--infer_upper_factor", type=float, default=1.0,
        help="Soft sampling upper bound = prior_high * factor (default: 1.0, i.e. no extension beyond prior_high)",
    )
    parser.add_argument(
        "--drift_third_pauli",
        type=str,
        default="ZI",
        help="Third Pauli term to plot in drift trajectory (default: ZI). Example: YY.",
    )
    args = parser.parse_args()

    run_experiment(
        results_dir=args.results_dir,
        num_posterior_samples=args.num_posterior_samples,
        n_runs=args.n_runs,
        out_dir=args.out_dir,
        drift_mode=args.drift_mode,
        drift_scale=args.drift_scale,
        drift_period=args.drift_period,
        drift_seed=args.drift_seed,
        theta_0_file=args.theta_0_file,
        simulator_backend=args.simulator_backend,
        infer_upper_factor=args.infer_upper_factor,
        drift_third_pauli=args.drift_third_pauli,
    )
