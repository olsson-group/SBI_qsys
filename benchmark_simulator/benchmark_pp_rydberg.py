"""Benchmark Pauli propagation (Rydberg) vs Aer statevector (exact).

Four panels (matching the style of `benchmark_pp_paulinoise.py`):
  (a) MAE(PP, Aer) of a single observable Z_1 Z_2 (Pauli ZZ on qubits 0,1)
      for weight truncation max_weight ∈ {1,2,3,4,5}. Only weight truncation (eps=0).
      Each point is the average over N_DISPLACEMENTS random displacement samples.
  (b) Wall time for panel (a) settings.
  (c) Z_1 Z_2 results for the scaling sweep, max_weight ∈ {1,2,3,4,5} (no exact reference).
  (d) Wall time for panel (c) settings.

Outputs:
  - benchmark_pp_rydberg.pdf
  - benchmark_pp_rydberg.json  (includes shared thetas + all results)
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Local imports (repo root on sys.path)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rydberg_aer_sim import _build_circuit, _build_interactions, _expect_pauli
from sbi_rydberg.lattice_simulate import (
    RydbergSimConfig,
    ideal_square_positions,
)
from sbi_rydberg.pauli_propagation_rydberg import simulate_batch_pauli_propagation


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

N_DISPLACEMENTS = 20
D_MAX_UM = 0.15  # prior ±150nm (matches `run_sbi_rydberg.py`)

MAX_WEIGHTS = [1, 2, 3, 4, 5]

N_COMP = [4, 9, 16]  # 2×2, 3×3, 4×4 (exact Aer reference feasible)
# Squares for scaling panels: 3^2..12^2 = 9..144
N_SCALE = [k * k for k in range(3, 13)]

BASE_RNG_SEED = 42

_HERE = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(_HERE, "benchmark_pp_rydberg.json")
OUT_PDF = os.path.join(_HERE, "benchmark_pp_rydberg.pdf")

# Prefer redrawing from saved JSON to avoid rerunning benchmarks.
LOAD_FROM_JSON = True
SHOW_FIG = False

# Single observable to benchmark.
OBSERVABLE = "Z1Z2"  # Pauli Z_0 Z_1 (shows weight-truncation convergence on |+...+>)

def redraw_from_json(
    *,
    json_path: str = JSON_PATH,
    out_pdf: str = OUT_PDF,
    show_fig: bool = SHOW_FIG,
) -> str:
    """Redraw PDF from saved JSON (no simulation). Returns out_pdf path."""
    with open(json_path, "r", encoding="utf-8") as f:
        _data = json.load(f)

    _meta = _data.get("meta", {})
    n_comp = np.array(_meta.get("N_COMP", N_COMP), dtype=float)
    n_scale = np.array(_meta.get("N_SCALE", N_SCALE), dtype=float)
    max_weights = list(_meta.get("MAX_WEIGHTS", MAX_WEIGHTS))

    res = _data.get("results", {})
    time_aer = {int(k): float(v) for k, v in res.get("time_exact", {}).items()}
    time_pp = {int(k): {int(w): float(tv) for w, tv in vv.items()} for k, vv in res.get("time_pp", {}).items()}
    mae = {int(k): {int(w): float(mv) for w, mv in vv.items()} for k, vv in res.get("mae", {}).items()}
    mean_x1 = {int(k): {int(w): float(mv) for w, mv in vv.items()} for k, vv in res.get("pp_mean_obs", {}).items()}

    mae_arr = np.array([[mae[int(n)][int(w)] for n in n_comp] for w in max_weights])
    texact_comp = np.array([time_aer[int(n)] for n in n_comp])
    tpp_comp = np.array([[time_pp[int(n)][int(w)] for n in n_comp] for w in max_weights])

    x1_scale = np.array([[mean_x1[int(n)][int(w)] for n in n_scale] for w in max_weights])
    tpp_scale = np.array([[time_pp[int(n)][int(w)] for n in n_scale] for w in max_weights])

    _RC = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 18,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 14,
        "axes.linewidth": 1.2,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }
    _TP = dict(direction="out", length=5, width=1.2)
    _TPm = dict(direction="out", length=3, width=0.9)

    # Colors for w=1..5: ensure w<=1 vs w<=2 are clearly distinct; keep w=5 green subdued.
    colors = ["#d38d95", "#e45756", "#7a9cc6", "#5dabce", "#3f8f6f"]
    c_exact = "#8f8f8f"
    markers = ["o", "s", "D", "^", "v"]
    _LW = 2.8
    _MS = 7.5

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(2, 2, figsize=(8.0, 8.4), constrained_layout=True)
        ax1, ax2, ax3, ax4 = axes.ravel()

        def _fmt_tick(x, _pos=None):
            # Use compact numeric labels:
            # 0.00 -> 0, 0.10 -> 0.1, 1e0 -> 1, etc.
            if abs(x) < 1e-15:
                return "0"
            if abs(x - 1.0) < 1e-12:
                return "1"
            return f"{x:g}"

        # (a) MAE
        for i, w in enumerate(max_weights):
            ax1.plot(n_comp, mae_arr[i], marker=markers[i], linestyle="-", lw=_LW, ms=_MS, color=colors[i], label=rf"$w \leq {w}$")
        ax1.set_xlabel("$n$")
        ax1.set_ylabel(r"$\langle Z_1 Z_2 \rangle$ error")
        ax1.set_xticks(list(n_comp.astype(int)))
        ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_tick))
        ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax1.tick_params(axis="both", which="major", **_TP)

        # (b) time (log y)
        ax2.plot(n_comp, texact_comp * 1e3, marker="P", linestyle="-", lw=_LW, ms=_MS, color=c_exact, label="Statevector")
        for i, w in enumerate(max_weights):
            ax2.plot(n_comp, tpp_comp[i] * 1e3, marker=markers[i], linestyle="-", lw=_LW, ms=_MS, color=colors[i], label=rf"PP $w \leq {w}$")
        ax2.set_xlabel("$n$")
        ax2.set_ylabel("Time [ms]")
        ax2.set_yscale("log")
        ax2.set_xticks(list(n_comp.astype(int)))
        ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
        ax2.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_tick))
        ax2.tick_params(axis="both", which="major", **_TP)
        ax2.tick_params(axis="both", which="minor", **_TPm)
        ax2.legend(frameon=False, loc="upper left", bbox_to_anchor=(-0.03, 1.02), fontsize=14)

        # (c) X1 mean
        for i, w in enumerate(max_weights):
            ax3.plot(n_scale, x1_scale[i], marker=markers[i], linestyle="-", lw=_LW, ms=_MS, color=colors[i], label=rf"$w \leq {w}$")
        ax3.set_xlabel("$n$")
        ax3.set_ylabel(r"$\langle Z_1 Z_2 \rangle$")
        ax3.set_xticks([10, 50, 90, 130])
        ax3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
        ax3.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_tick))
        ax3.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax3.tick_params(axis="both", which="major", **_TP)

        # (d) time for (c) (log y)
        for i, w in enumerate(max_weights):
            ax4.plot(n_scale, tpp_scale[i] * 1e3, marker=markers[i], linestyle="-", lw=_LW, ms=_MS, color=colors[i], label=rf"$w \leq {w}$")
        ax4.set_xlabel("$n$")
        ax4.set_ylabel("Time [ms]")
        ax4.set_yscale("log")
        ax4.set_xticks([10, 50, 90, 130])
        ax4.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
        ax4.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_tick))
        ax4.tick_params(axis="both", which="major", **_TP)
        ax4.tick_params(axis="both", which="minor", **_TPm)

        fig.savefig(out_pdf)
        if show_fig:
            plt.show()
    return out_pdf


# Use the same physical parameters as `run_sbi_rydberg.py` default config.
# For exact-vs-PP comparisons we vary small (nx, ny) so that Aer statevector remains feasible.
BASE_CFG_KW = dict(
    lattice_constant_um=8.0,
    omega_rad_per_us=15.0,
    detuning_rad_per_us=15.0,
    times_us=(0.01,),  # single snapshot
    dt_us=0.01,
    interaction_scale=1.0,
    v_threshold=1.0,
)


def _to_jsonable(x):
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x


def _layout_for_n(n_atoms: int) -> tuple[int, int]:
    """Pick a near-square (nx, ny) with nx*ny == n_atoms when possible.

    For primes, fall back to a 1D chain (nx=n, ny=1).
    """
    best = None
    for ny in range(1, int(np.sqrt(n_atoms)) + 1):
        if n_atoms % ny != 0:
            continue
        nx = n_atoms // ny
        cand = (nx, ny) if nx >= ny else (ny, nx)
        score = cand[0] - cand[1]  # minimize aspect ratio
        if best is None or score < best[0]:
            best = (score, cand[0], cand[1])
    if best is None:
        return int(n_atoms), 1
    return int(best[1]), int(best[2])


def _cfg_for_n(n_atoms: int) -> RydbergSimConfig:
    n_atoms = int(n_atoms)
    if n_atoms == 4:
        return RydbergSimConfig(nx=2, ny=2, **BASE_CFG_KW)
    if n_atoms == 9:
        return RydbergSimConfig(nx=3, ny=3, **BASE_CFG_KW)
    if n_atoms == 16:
        return RydbergSimConfig(nx=4, ny=4, **BASE_CFG_KW)
    nx, ny = _layout_for_n(n_atoms)
    return RydbergSimConfig(nx=nx, ny=ny, **BASE_CFG_KW)


def _obs_z1z2(cfg: RydbergSimConfig) -> list[np.ndarray]:
    row = np.zeros(cfg.n_atoms, dtype=np.uint8)
    row[0] = 3  # Z on qubit 0
    row[1] = 3  # Z on qubit 1
    return [row]


def _aer_expect_z1z2(theta_flat: np.ndarray, cfg: RydbergSimConfig) -> float:
    """Exact Aer statevector expectation of Z_0 Z_1 at the final snapshot."""
    from qiskit_aer import AerSimulator

    n = cfg.n_atoms
    th = np.asarray(theta_flat, dtype=np.float64).reshape(-1)
    ideal = ideal_square_positions(cfg.nx, cfg.ny, cfg.lattice_constant_um)
    pos = ideal + th.reshape(n, 2)
    interactions = _build_interactions(pos, n, cfg.c6_rad_um6_per_us, cfg.interaction_scale, cfg.v_threshold)

    times = sorted(float(t) for t in cfg.times_us)
    n_steps = int(np.ceil(times[-1] / cfg.dt_us))
    snap_steps = [int(round(t / cfg.dt_us)) for t in times]

    qc = _build_circuit(
        n,
        float(cfg.omega_rad_per_us),
        float(cfg.detuning_rad_per_us),
        interactions,
        float(cfg.dt_us),
        n_steps,
        snap_steps,
        z_perturb=None,
    )

    gpu_available = "GPU" in AerSimulator().available_devices()
    sim = AerSimulator(method="statevector", device="GPU" if gpu_available else "CPU")
    result = sim.run([qc], shots=1).result()
    sv = np.asarray(result.data(0)[f"sv_{snap_steps[-1]}"])
    return float(_expect_pauli(sv, [(0, "Z"), (1, "Z")], 2**n))


def _benchmark_exact_and_pp_for_n(
    n: int,
    thetas: np.ndarray,  # (N_DISPLACEMENTS, 2*n)
    max_weights: list[int],
) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    """Return (mae_by_w, t_pp_by_w, t_exact). Times are averaged over displacements."""
    cfg = _cfg_for_n(n)
    obs = _obs_z1z2(cfg)
    col = 0

    # Warm-up: avoid counting one-time initialization overheads (notably Aer).
    # We warm up both exact and PP once for this n.
    _ = _aer_expect_z1z2(thetas[0], cfg)
    _ = simulate_batch_pauli_propagation(
        thetas[0].reshape(1, -1),
        cfg,
        eps=0.0,
        max_weight=int(max_weights[-1]),
        observables=obs,
        device="cpu",
        verbose=False,
        chunk_size=512,
    )

    # Exact (Aer statevector) once per theta sample.
    t_exact = []
    exact_vals = []
    for th in thetas:
        t0 = time.perf_counter()
        x = _aer_expect_z1z2(th, cfg)
        t_exact.append(time.perf_counter() - t0)
        exact_vals.append(float(x))
    exact_vals = np.asarray(exact_vals, dtype=np.float64)
    t_exact_mean = float(np.mean(t_exact))

    mae_by_w: dict[int, float] = {}
    t_pp_by_w: dict[int, float] = {}

    for w in max_weights:
        t_pp = []
        pp_vals = []
        for th in thetas:
            t0 = time.perf_counter()
            xpp = simulate_batch_pauli_propagation(
                th.reshape(1, -1),
                cfg,
                eps=0.0,            # only weight truncation
                max_weight=int(w),
                observables=obs,
                device="cpu",
                verbose=False,
                chunk_size=512,
            )[0]
            t_pp.append(time.perf_counter() - t0)
            pp_vals.append(float(xpp[col]))
        pp_vals = np.asarray(pp_vals, dtype=np.float64)
        mae_by_w[w] = float(np.mean(np.abs(pp_vals - exact_vals)))
        t_pp_by_w[w] = float(np.mean(t_pp))

    return mae_by_w, t_pp_by_w, {0: t_exact_mean}


def _benchmark_pp_scaling(
    n: int,
    thetas: np.ndarray,  # (N_DISPLACEMENTS, 2*n)
    max_weights: list[int],
) -> tuple[dict[int, float], dict[int, float]]:
    """Return (mean_zz_by_w, t_pp_by_w) averaged over displacements."""
    cfg = _cfg_for_n(n)
    obs = _obs_z1z2(cfg)
    col = 0

    mean_x_by_w: dict[int, float] = {}
    t_pp_by_w: dict[int, float] = {}

    for w in max_weights:
        vals = []
        times = []
        for th in thetas:
            t0 = time.perf_counter()
            xpp = simulate_batch_pauli_propagation(
                th.reshape(1, -1),
                cfg,
                eps=0.0,
                max_weight=int(w),
                observables=obs,
                device="cpu",
                verbose=False,
                chunk_size=512,
            )[0]
            times.append(time.perf_counter() - t0)
            vals.append(float(xpp[col]))
        mean_x_by_w[w] = float(np.mean(vals))
        t_pp_by_w[w] = float(np.mean(times))

    return mean_x_by_w, t_pp_by_w


def main() -> None:
    if LOAD_FROM_JSON and os.path.exists(JSON_PATH):
        try:
            redraw_from_json(json_path=JSON_PATH, out_pdf=OUT_PDF, show_fig=SHOW_FIG)
            print(f"Saved → {OUT_PDF}", flush=True)
            return
        except Exception:
            # JSON exists but doesn't match the current plotting expectations (e.g. different observable).
            # Fall through to rerun benchmarks and overwrite JSON.
            pass
    rng = np.random.default_rng(BASE_RNG_SEED)
    shared_thetas: dict[int, np.ndarray] = {}
    for n in sorted(set(N_COMP + N_SCALE)):
        shared_thetas[n] = rng.uniform(-D_MAX_UM, D_MAX_UM, size=(N_DISPLACEMENTS, 2 * n)).astype(np.float32)

    # Results containers (JSON-friendly)
    results = {
        "mae": {},             # n -> w -> mae
        "time_pp": {},         # n -> w -> seconds
        "time_exact": {},      # n -> seconds
        "pp_mean_obs": {},     # n -> w -> mean value
    }

    # (a)(b): exact + pp
    print("Exact (Aer statevector) vs Pauli propagation, Z_1Z_2, averaged over displacements.")
    for n in N_COMP:
        mae_by_w, t_pp_by_w, t_exact = _benchmark_exact_and_pp_for_n(n, shared_thetas[n], MAX_WEIGHTS)
        results["mae"][str(n)] = {str(w): mae_by_w[w] for w in MAX_WEIGHTS}
        results["time_pp"][str(n)] = {str(w): t_pp_by_w[w] for w in MAX_WEIGHTS}
        results["time_exact"][str(n)] = float(t_exact[0])
        mae_str = "  ".join(f"w{w}:{mae_by_w[w]:.2e}" for w in MAX_WEIGHTS)
        tpp_str = "  ".join(f"w{w}:{t_pp_by_w[w]*1e3:.1f}ms" for w in MAX_WEIGHTS)
        print(f"  n={n:3d}  exact={t_exact[0]*1e3:7.1f}ms   MAE[{mae_str}]   PP[{tpp_str}]", flush=True)

    # (c)(d): pp only, scaling sweep
    print("\nPauli propagation scaling (no exact), Z_1Z_2 results and time.")
    for n in N_SCALE:
        mean_x_by_w, t_pp_by_w = _benchmark_pp_scaling(n, shared_thetas[n], MAX_WEIGHTS)
        results["pp_mean_obs"][str(n)] = {str(w): mean_x_by_w[w] for w in MAX_WEIGHTS}
        results["time_pp"][str(n)] = {str(w): t_pp_by_w[w] for w in MAX_WEIGHTS}
        v_str = "  ".join(f"w{w}:{mean_x_by_w[w]:+.3f}" for w in MAX_WEIGHTS)
        t_str = "  ".join(f"w{w}:{t_pp_by_w[w]*1e3:.1f}ms" for w in MAX_WEIGHTS)
        print(f"  n={n:3d}  X1[{v_str}]  time[{t_str}]", flush=True)

    meta = {
        "script": os.path.basename(__file__),
        "BASE_RNG_SEED": int(BASE_RNG_SEED),
        "N_DISPLACEMENTS": int(N_DISPLACEMENTS),
        "D_MAX_UM": float(D_MAX_UM),
        "MAX_WEIGHTS": list(MAX_WEIGHTS),
        "N_COMP": list(N_COMP),
        "N_SCALE": list(N_SCALE),
            "cfg_base": BASE_CFG_KW,
            "note": "Only weight truncation for PP (eps=0). Observable is Z_1 Z_2 (Pauli ZZ on qubits 0,1).",
        }

    data_out = {
        "meta": meta,
        "shared_thetas": shared_thetas,
        "results": results,
        "cfg_example_n14": asdict(_cfg_for_n(14)),
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(data_out), f, indent=2)
    print(f"\nSaved → {JSON_PATH}", flush=True)

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------

    n_comp = np.array(N_COMP, dtype=float)
    n_scale = np.array(N_SCALE, dtype=float)

    # Build arrays: shape (len(MAX_WEIGHTS), len(N_*))
    mae_arr = np.array([[float(results["mae"][str(n)][str(w)]) for n in N_COMP] for w in MAX_WEIGHTS])
    tpp_comp = np.array([[float(results["time_pp"][str(n)][str(w)]) for n in N_COMP] for w in MAX_WEIGHTS])
    texact_comp = np.array([float(results["time_exact"][str(n)]) for n in N_COMP])

    x1_scale = np.array([[float(results["pp_mean_obs"][str(n)][str(w)]) for n in N_SCALE] for w in MAX_WEIGHTS])
    tpp_scale = np.array([[float(results["time_pp"][str(n)][str(w)]) for n in N_SCALE] for w in MAX_WEIGHTS])

    _RC = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 18,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 14,
        "axes.linewidth": 1.2,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }
    _TP = dict(direction="out", length=5, width=1.2)
    _TPm = dict(direction="out", length=3, width=0.9)

    def _fmt_tick(x, _pos=None):
        # Compact numeric labels: 0.00 -> 0, 0.20 -> 0.2, 1.0 -> 1.
        if abs(x) < 1e-15:
            return "0"
        if abs(x - 1.0) < 1e-12:
            return "1"
        return f"{x:g}"

    # Simple, high-contrast palette (5 truncations + exact baseline)
    # Colors for w=1..5: keep a cohesive palette but ensure w=2,3 are not too faint.
    # w=5 uses a brighter "dopamine" green.
    colors = ["#d38d95", "#f28e8e", "#7fb7d8", "#5dabce", "#00c853"]
    c_exact = "#6b6b6b"

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(2, 2, figsize=(8.0, 8.4), constrained_layout=True)
        ax1, ax2, ax3, ax4 = axes.ravel()

        # (a) MAE
        for i, w in enumerate(MAX_WEIGHTS):
            ax1.plot(n_comp, mae_arr[i], "o-", lw=2.6, ms=7.0, color=colors[i], label=rf"$w \leq {w}$")
        ax1.set_xlabel("$n$")
        ax1.set_ylabel(r"$\langle Z_1 Z_2 \rangle$ error")
        ax1.set_xticks([4, 9, 16])
        ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_tick))
        ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax1.tick_params(axis="both", which="major", **_TP)

        # (b) time (log y)
        ax2.plot(n_comp, texact_comp * 1e3, "s-", lw=2.6, ms=7.0, color=c_exact, label="Statevector")
        for i, w in enumerate(MAX_WEIGHTS):
            ax2.plot(n_comp, tpp_comp[i] * 1e3, "o-", lw=2.6, ms=7.0, color=colors[i], label=rf"PP $w \leq {w}$")
        ax2.set_xlabel("$n$")
        ax2.set_ylabel("Time [ms]")
        ax2.set_yscale("log")
        ax2.set_xticks([4, 9, 16])
        ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
        ax2.tick_params(axis="both", which="major", **_TP)
        ax2.tick_params(axis="both", which="minor", **_TPm)
        ax2.legend(frameon=False, loc="upper left", bbox_to_anchor=(-0.03, 1.02), fontsize=14)

        # (c) X1 results (mean over 5 displacements)
        for i, w in enumerate(MAX_WEIGHTS):
            ax3.plot(n_scale, x1_scale[i], "o-", lw=2.6, ms=7.0, color=colors[i], label=rf"$w \leq {w}$")
        ax3.set_xlabel("$n$")
        ax3.set_ylabel(r"$\langle Z_1 Z_2 \rangle$")
        ax3.set_xticks([10, 50, 90, 130])
        ax3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
        ax3.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_tick))
        ax3.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax3.tick_params(axis="both", which="major", **_TP)

        # (d) time for (c) (log y)
        for i, w in enumerate(MAX_WEIGHTS):
            ax4.plot(n_scale, tpp_scale[i] * 1e3, "o-", lw=2.6, ms=7.0, color=colors[i], label=rf"$w \leq {w}$")
        ax4.set_xlabel("$n$")
        ax4.set_ylabel("Time [ms]")
        ax4.set_yscale("log")
        ax4.set_xticks([10, 50, 90, 130])
        ax4.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
        ax4.tick_params(axis="both", which="major", **_TP)
        ax4.tick_params(axis="both", which="minor", **_TPm)

        fig.savefig(OUT_PDF)
        print(f"Saved → {OUT_PDF}", flush=True)
        if SHOW_FIG:
            plt.show()


if __name__ == "__main__":
    main()

