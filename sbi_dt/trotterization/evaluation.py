"""Evaluate the Trotter denoiser against the noisy baseline and ZNE; plot trotter_results.pdf."""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

from .circuit import build_neel_trotter_circuit, compute_ideal_expectations
from .runner import TrotterNoisyRunner

plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.fontset"] = "dejavusans"

# Figure styles for trotter_results.pdf (colours are shared)
STYLE_DT = dict(
    figsize=(12, 8.5), fs_label=26, fs_tick=26, fs_xtick_bar=24, fs_ratio=22, lw=3.0,
    tick_kw=dict(length=8, width=1.4), y_nbins=3, coupling=r"$j$", ratio_split=True,
    legend_fs=24, legend_y=0.01,
    legend_kw=dict(handlelength=1.0, handletextpad=0.35, columnspacing=0.9, labelspacing=0.25),
)
_BAR_RATIO_HALFGAP = 0.02   # split ratio label: half-gap between the digits and "×"

_C_NOISY, _C_DENOISED, _C_IDEAL = "#D96B64", "#8B65C8", "#2C3E50"
_C_NOISY_BAR, _C_DENOISED_BAR = "#E8A09A", "#A98FD4"
_ZNE_COLORS = {"linear": "#5A9EC8", "polynomial": "#4DB8A8", "exponential": "#C2607A"}
_ZNE_COLORS_BAR = {"linear": "#7BAFD4", "polynomial": "#7EC4BA", "exponential": "#C2607A"}
_ZNE_LABELS = {"linear": "ZNE (lin)", "polynomial": "ZNE (quad)", "exponential": "ZNE (exp)"}
_ZNE_TICK_LABELS = {"linear": "(lin)", "polynomial": "(quad)", "exponential": "(exp)"}


def _tick_fmt(x: float, pos) -> str:
    """Tick label: whole numbers without decimals."""
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:g}"


def _avg_x_mag(data: np.ndarray, n_qubits: int) -> np.ndarray:
    """M_x(t) = mean_i ⟨X_i⟩ (observables are ordered X_i, then Z_i Z_{i+1})."""
    return np.mean(data[:, :n_qubits], axis=1)


def _avg_zz(data: np.ndarray, n_qubits: int) -> np.ndarray:
    """C_ZZ(t) = mean_i ⟨Z_i Z_{i+1}⟩."""
    return np.mean(data[:, n_qubits:], axis=1)


def _zne_extrapolate(
    scale_factors: List[int],
    results_by_scale: List[np.ndarray],
    method: str = "polynomial",
) -> np.ndarray:
    """Extrapolate expectations measured at noise scale factors to zero noise.

    results_by_scale: one (T, n_obs) array per scale factor. Methods: "linear" (linear fit),
    "polynomial" (quadratic through the points), "exponential" (A·exp(Bλ) fit, falls back to
    polynomial when the values change sign or vanish).
    Returns a (T, n_obs) array.
    """
    x = np.array(scale_factors, dtype=float)
    stacked = np.stack(results_by_scale, axis=0)   # (n_scales, T, n_obs)
    T, n_obs = stacked.shape[1], stacked.shape[2]
    out = np.zeros((T, n_obs), dtype=np.float32)
    deg = min(len(scale_factors) - 1, 2)

    for t in range(T):
        for o in range(n_obs):
            y = stacked[:, t, o].astype(float)
            if method == "exponential":
                signs = np.sign(y)
                if np.all(signs == signs[0]) and np.all(np.abs(y) > 1e-12):
                    _, log_A = np.polyfit(x, np.log(np.abs(y)), 1)
                    out[t, o] = float(signs[0] * np.exp(log_A))
                else:
                    out[t, o] = float(np.polyval(np.polyfit(x, y, deg), 0.0))
            elif method == "linear":
                out[t, o] = float(np.polyval(np.polyfit(x, y, 1), 0.0))
            else:
                out[t, o] = float(np.polyval(np.polyfit(x, y, deg), 0.0))
    return out


def eval_ideal_noisy_zne(
    runner: TrotterNoisyRunner,
    n_qubits: int,
    J: float,
    h: float,
    dt: float,
    n_steps_list: List[int],
    observables: List[str],
    zne_scale_factors: Optional[List[int]] = None,
    zne_methods: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Ideal, noisy and ZNE expectations of the Néel Trotter circuits at coupling J, per depth.

    Returns (ideal, noisy, zne) with ideal/noisy of shape (T, n_obs) and zne a dict
    method -> (T, n_obs), empty when ZNE is disabled.
    """
    circuits = [build_neel_trotter_circuit(n_qubits, J, h, dt, k) for k in n_steps_list]
    ideal = np.array([compute_ideal_expectations(qc, observables) for qc in circuits], dtype=np.float32)
    noisy = np.array(runner.run_batch(circuits, observables), dtype=np.float32)

    zne: Dict[str, np.ndarray] = {}
    methods = list(zne_methods) if zne_methods else []
    if zne_scale_factors is not None and methods:
        other = [lam for lam in zne_scale_factors if lam != 1]
        print(f"    ZNE: running scale factors {other} (scale=1 reused), methods={methods} ...")
        # scale 1 is the noisy run itself
        by_scale = [
            noisy if lam == 1 else np.array(runner.run_batch_scaled(circuits, observables, lam), dtype=np.float32)
            for lam in zne_scale_factors
        ]
        for m in methods:
            if m in ("linear", "polynomial", "exponential"):
                zne[m] = _zne_extrapolate(zne_scale_factors, by_scale, method=m)
    return ideal, noisy, zne


def mae_per_depth(ideal: list, estimate: list) -> np.ndarray:
    """Mean absolute error over observables for every (J, depth): shape (n_J, T)."""
    return np.array([np.mean(np.abs(np.asarray(e) - np.asarray(i)), axis=1) for i, e in zip(ideal, estimate)],
                    dtype=np.float64)


def plot_results(results: dict, output_dir: str, style: Optional[dict] = None) -> str:
    """Draw trotter_results.pdf (M_x, C_ZZ, MAE vs t, overall MAE bars) from a results dict."""
    st = style or STYLE_DT
    lw, fs_label, fs_tick = st["lw"], st["fs_label"], st["fs_tick"]
    coupling = st["coupling"]

    def style_axes(ax):
        ax.tick_params(axis="both", labelsize=fs_tick, **st["tick_kw"])
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=st["y_nbins"]))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_tick_fmt))

    cfg = results["config"]
    t_eval = np.array(results["t_eval"])
    all_ideal = [np.array(x) for x in results["ideal"]]
    all_noisy = [np.array(x) for x in results["noisy"]]
    all_denoised = [np.array(x) for x in results["denoised"]]
    methods = [m for m in results["mae"].get("zne_methods", []) if f"zne_{m}" in results]
    all_zne = {m: [np.array(x) for x in results[f"zne_{m}"]] for m in methods}
    use_zne = bool(methods)

    n_qubits = cfg["n_qubits"]
    t_boundary = max(cfg["n_steps_train"]) * cfg["dt"]

    mae_noisy = mae_per_depth(all_ideal, all_noisy)
    mae_denoised = mae_per_depth(all_ideal, all_denoised)
    mae_zne = {m: mae_per_depth(all_ideal, all_zne[m]) for m in methods}
    mae_n_mean, mae_n_std = mae_noisy.mean(0), mae_noisy.std(0)
    mae_d_mean, mae_d_std = mae_denoised.mean(0), mae_denoised.std(0)

    # Overall MAE bars, ordered Noisy, ZNE…, SBI-DT
    mean_n = float(mae_n_mean.mean())
    bar_means = [mean_n]
    bar_stds = [float(mae_noisy.mean(axis=0).std())]
    bar_labels = ["Noisy"]
    bar_colors = [_C_NOISY_BAR]
    for m in methods:
        bar_means.append(float(mae_zne[m].mean(0).mean()))
        bar_stds.append(float(mae_zne[m].mean(axis=0).std()))
        bar_labels.append("ZNE\n" + _ZNE_TICK_LABELS.get(m, f"({m})"))
        bar_colors.append(_ZNE_COLORS_BAR.get(m, "gray"))
    bar_means.append(float(mae_d_mean.mean()))
    bar_stds.append(float(mae_denoised.mean(axis=0).std()))
    bar_labels.append("SBI-DT")
    bar_colors.append(_C_DENOISED_BAR)

    fig, axes = plt.subplots(2, 2, figsize=st["figsize"])
    axes[0, 1].sharex(axes[0, 0])

    # (a) M_x and (b) C_ZZ, mean over J
    for ax_idx, (extract_fn, ylabel) in enumerate([(_avg_x_mag, r"$M_x$"), (_avg_zz, r"$C_{ZZ}$")]):
        ax = axes[0, ax_idx]
        def mean_curve(arrs):
            return np.array([extract_fn(a, n_qubits) for a in arrs]).mean(0)
        ax.plot(t_eval, mean_curve(all_ideal), color=_C_IDEAL, lw=lw, zorder=3)
        ax.plot(t_eval, mean_curve(all_noisy), color=_C_NOISY, lw=lw, zorder=4)
        for m in methods:
            ax.plot(t_eval, mean_curve(all_zne[m]), color=_ZNE_COLORS.get(m, "gray"), lw=lw, zorder=5)
        ax.plot(t_eval, mean_curve(all_denoised), color=_C_DENOISED, lw=lw, zorder=6)
        ax.axvline(t_boundary, color="#888888", lw=1.2, ls=":", zorder=0)
        ax.set_xlabel("$t$", fontsize=fs_label)
        ax.set_ylabel(ylabel, fontsize=fs_label)
        style_axes(ax)
        ax.grid(False)

    # (c) MAE vs t
    ax = axes[1, 0]
    ax.fill_between(t_eval, np.maximum(mae_n_mean - mae_n_std, 0), mae_n_mean + mae_n_std,
                    alpha=0.15, color=_C_NOISY, zorder=1)
    for m in methods:
        zm, zs = mae_zne[m].mean(0), mae_zne[m].std(0)
        ax.fill_between(t_eval, np.maximum(zm - zs, 0), zm + zs,
                        alpha=0.15, color=_ZNE_COLORS.get(m, "gray"), zorder=2)
    ax.fill_between(t_eval, np.maximum(mae_d_mean - mae_d_std, 0), mae_d_mean + mae_d_std,
                    alpha=0.15, color=_C_DENOISED, zorder=3)
    ax.plot(t_eval, mae_n_mean, color=_C_NOISY, lw=lw, zorder=4)
    for m in methods:
        ax.plot(t_eval, mae_zne[m].mean(0), color=_ZNE_COLORS.get(m, "gray"), lw=lw, zorder=5)
    ax.plot(t_eval, mae_d_mean, color=_C_DENOISED, lw=lw, zorder=6)
    ax.axvline(t_boundary, color="#888888", lw=1.2, ls=":", zorder=0)
    ax.set_xlabel("$t$", fontsize=fs_label)
    ax.set_ylabel(f"MAE (over {coupling})", fontsize=fs_label)
    ax.set_ylim(bottom=0)
    y_top = ax.get_ylim()[1]
    ax.set_ylim(bottom=-0.02 * y_top, top=y_top)
    style_axes(ax)
    ax.xaxis.set_major_locator(mticker.FixedLocator([0, 1, 2, 3]))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_tick_fmt))
    ax.grid(False)

    # (d) overall MAE
    ax = axes[1, 1]
    x_pos = np.arange(len(bar_labels))
    ax.bar(x_pos, bar_means, yerr=bar_stds, width=0.65, capsize=3, color=bar_colors,
           edgecolor=_C_IDEAL, linewidth=1.4, error_kw=dict(ecolor=_C_IDEAL, elinewidth=1.4))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bar_labels, fontsize=st["fs_xtick_bar"])
    ax.set_ylabel(f"MAE (over {coupling}, $t$)", fontsize=fs_label)
    ax.set_ylim(bottom=0)
    for i in range(1, len(bar_means)):
        if bar_means[i] > 0:
            ratio = mean_n / bar_means[i]
            if st["ratio_split"]:
                ax.text(x_pos[i] - _BAR_RATIO_HALFGAP, bar_means[i], f"{ratio:.1f}",
                        ha="right", va="bottom", fontsize=st["fs_ratio"], color="black")
                ax.text(x_pos[i] + _BAR_RATIO_HALFGAP, bar_means[i], r"$\times$",
                        ha="left", va="bottom", fontsize=st["fs_ratio"], color="black")
            else:
                ax.text(x_pos[i] + 0.05, bar_means[i], f"{ratio:.1f}$\\times$",
                        ha="left", va="bottom", fontsize=st["fs_ratio"], color="black")
    style_axes(ax)
    ax.grid(False)

    handles = [Line2D([0], [0], color=_C_IDEAL, lw=lw, label="Ideal"),
               Line2D([0], [0], color=_C_NOISY, lw=lw, label="Noisy")]
    for m in methods:
        handles.append(Line2D([0], [0], color=_ZNE_COLORS.get(m, "gray"), lw=lw,
                              label=_ZNE_LABELS.get(m, f"ZNE ({m})")))
    handles.append(Line2D([0], [0], color=_C_DENOISED, lw=lw, label="SBI-DT"))
    fig.legend(handles=handles, loc="lower center", ncol=3 + len(methods) if use_zne else 4, frameon=False,
               fontsize=st["legend_fs"], bbox_to_anchor=(0.5, st["legend_y"]), **st["legend_kw"])

    plt.tight_layout(rect=[0, 0.10, 1, 1])
    out_path = os.path.join(output_dir, "trotter_results.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def summarize_mae(
    all_ideal: list, all_noisy: list, all_denoised: list, all_zne: Dict[str, list],
    methods: List[str], zne_scale_factors: Optional[List[int]],
) -> dict:
    """MAE summary stored in the results JSON (per-depth means, overall values, improvement factors)."""
    mae_n = np.abs(mae_per_depth(all_ideal, all_noisy).mean(0))
    mae_d = np.abs(mae_per_depth(all_ideal, all_denoised).mean(0))
    mean_n, mean_d = float(mae_n.mean()), float(mae_d.mean())
    entry = {
        "noisy_mean": mae_n.tolist(),
        "denoised_mean": mae_d.tolist(),
        "overall_noisy": mean_n,
        "overall_denoised": mean_d,
        "improvement_denoised": mean_n / mean_d if mean_d > 0 else float("inf"),
    }
    print(f"\n  MAE (mean over depths & j): Noisy={mean_n:.6f}  Denoised={mean_d:.6f}  "
          f"Improvement={entry['improvement_denoised']:.2f}×")
    for m in methods:
        mae_z = np.abs(mae_per_depth(all_ideal, all_zne[m]).mean(0))
        mean_z = float(mae_z.mean())
        entry[f"zne_{m}_mean"] = mae_z.tolist()
        entry[f"overall_zne_{m}"] = mean_z
        entry[f"improvement_zne_{m}"] = mean_n / mean_z if mean_z > 0 else float("inf")
        print(f"  MAE {_ZNE_LABELS.get(m, m)} {list(zne_scale_factors)}: {mean_z:.6f}  "
              f"Improvement={entry[f'improvement_zne_{m}']:.2f}×")
    if methods:
        entry["zne_scale_factors"] = list(zne_scale_factors)
        entry["zne_methods"] = list(methods)
    return entry


def save_results(
    output_dir: str, config: Optional[dict], J_values: list, t_eval: np.ndarray,
    all_ideal: list, all_noisy: list, all_denoised: list, all_zne: Dict[str, list],
    methods: List[str], zne_scale_factors: Optional[List[int]], style: Optional[dict] = None,
) -> dict:
    """Write trotter_dynamics_results.json and trotter_results.pdf; returns the results dict."""
    cfg_serial = {k: (list(v) if isinstance(v, tuple) else v) for k, v in (config or {}).items()}
    results = {
        "config": cfg_serial,
        "J_values": J_values,
        "t_eval": t_eval.tolist(),
        "ideal": [x.tolist() for x in all_ideal],
        "noisy": [x.tolist() for x in all_noisy],
        "denoised": [x.tolist() for x in all_denoised],
    }
    for m in methods:
        results[f"zne_{m}"] = [x.tolist() for x in all_zne[m]]
    results["mae"] = summarize_mae(all_ideal, all_noisy, all_denoised, all_zne, methods, zne_scale_factors)
    results["timestamp"] = datetime.now().isoformat()
    plot_results(results, output_dir, style)
    print("  Saved: trotter_results.pdf")
    result_path = os.path.join(output_dir, "trotter_dynamics_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {result_path}")
    return results


def evaluate_and_plot(
    model: nn.Module,
    noise_params: np.ndarray,
    n_qubits: int,
    h: float,
    dt: float,
    n_steps_train: List[int],
    n_steps_eval: List[int],
    observables: List[str],
    J_eval: Optional[List[float]] = None,
    output_dir: str = "trotter_dynamics_results",
    random_seed: int = 42,
    device: str = "cpu",
    cx_topology: str = "brickwall",
    config: Optional[dict] = None,
    zne_scale_factors: Optional[List[int]] = None,
    zne_methods: Optional[List[str]] = None,
    aer_device: str = "auto",
    aer_max_parallel_threads: Optional[int] = None,
    aer_max_parallel_experiments: Optional[int] = None,
    aer_executor_workers: Optional[int] = None,
    aer_max_job_size: Optional[int] = None,
    gpu_batch_size: int = 1,
) -> None:
    """Compare denoised, noisy and ZNE expectations on held-out J values and extra depths.

    noise_params is the true noise that the TrotterNoisyRunner applies. Writes
    trotter_dynamics_results.json and trotter_results.pdf to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    J_values = list(J_eval) if J_eval else []
    methods = list(zne_methods) if zne_methods else []

    print(f"  Creating shared TrotterNoisyRunner (Aer, device={aer_device!r}) ...")
    runner = TrotterNoisyRunner(
        n_qubits, noise_params, random_seed=random_seed,
        cx_topology=cx_topology, aer_device=aer_device,
        aer_max_parallel_threads=aer_max_parallel_threads,
        aer_max_parallel_experiments=aer_max_parallel_experiments,
        aer_executor_workers=aer_executor_workers,
        aer_max_job_size=aer_max_job_size,
        gpu_batch_size=gpu_batch_size,
    )

    steps = torch.tensor(n_steps_eval, dtype=torch.float32, device=device)
    all_ideal, all_noisy, all_denoised = [], [], []
    all_zne: Dict[str, list] = {m: [] for m in methods}
    for j_i, J in enumerate(J_values):
        print(f"  Evaluating J={J:.3f} ({j_i+1}/{len(J_values)}) ...")
        ideal, noisy, zne = eval_ideal_noisy_zne(
            runner, n_qubits, J, h, dt, n_steps_eval, observables, zne_scale_factors, methods)
        with torch.no_grad():
            denoised = model(torch.tensor(noisy, device=device), steps).cpu().numpy()
        all_ideal.append(ideal)
        all_noisy.append(noisy)
        all_denoised.append(denoised)
        for m in methods:
            all_zne[m].append(zne[m])

    methods = [m for m in methods if all_zne[m]]
    save_results(output_dir, config, J_values, np.array(n_steps_eval, dtype=float) * dt,
                 all_ideal, all_noisy, all_denoised, all_zne, methods, zne_scale_factors)


def reproduce(results_dir: str, style: Optional[dict] = None) -> None:
    """Redraw trotter_results.pdf from the trotter_dynamics_results.json in results_dir."""
    results_dir = os.path.abspath(results_dir)
    with open(os.path.join(results_dir, "trotter_dynamics_results.json")) as f:
        results = json.load(f)
    print(f"  Saved plot to {plot_results(results, results_dir, style)}")
