"""Benchmark paulinoise_simulator_2 (Heisenberg) vs paulinoise_simulator_1 (Aer density matrix).

Three panels:
  (a) MAE(sim1, sim2) for n=2..14
  (b) Wall time sim1 vs sim2 for n=2..14, log y
  (c) sim2 scaling n=10..100: single Pauli Z_1Z_2 vs 1&2-local, log-log

All panels share the same fixed random params per n.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)  # repo root: ``sbi_paulinoise`` package, reference PDF for the palette
sys.path.insert(0, _ROOT)

N_SEEDS      = 5
NOISE_HIGH   = 1e-3
TOPOLOGY     = "linear"
WEIGHTS      = [1, 2]

N_COMP  = list(range(2, 15))        # n=2..14
# Log-spaced sizes for panels (c)/(d): ~6 points per decade, clean on log-log.
N_SCALE = sorted(set(int(round(x)) for x in np.logspace(np.log10(10), np.log10(1000), 13)))
N_ALL   = sorted(set(N_COMP + N_SCALE))

# Prefer redrawing from saved JSON to avoid rerunning benchmarks.
JSON_PATH = os.path.join(_HERE, "benchmark_pp_paulinoise.json")
LOAD_FROM_JSON = True
SHOW_FIG = False

# Extract a palette from the reference landscape PDF's colorbar.
def _extract_colorbar_palette(reference_pdf: str, n: int = 4) -> list[str] | None:
    """Return n hex colors sampled from the reference PDF colorbar (top→bottom).

    This uses a robust heuristic: render the first page and find the column with
    the strongest monotonic (R-B) gradient, then sample representative colors.
    """
    try:
        import fitz  # PyMuPDF
    except Exception:
        return None

    try:
        doc = fitz.open(reference_pdf)
        page = doc[0]
        mat = fitz.Matrix(3, 3)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3).astype(np.float32)
    except Exception:
        return None

    H, W, _ = img.shape
    # Focus on plot+colorbar area (exclude big margins)
    y0, y1 = int(H * 0.12), int(H * 0.82)
    core = img[y0:y1]
    Hc = core.shape[0]

    ys = np.arange(Hc, dtype=np.float32)
    ys = (ys - ys.mean()) / (ys.std() + 1e-6)

    best_x = None
    best_score = -1.0
    for x in range(int(W * 0.55), W):
        col = core[:, x, :]
        bright = col.mean(axis=1)
        if np.quantile(bright, 0.9) > 250 and np.quantile(bright, 0.1) > 240:
            continue
        sig = col[:, 0] - col[:, 2]  # R - B
        sig = (sig - sig.mean()) / (sig.std() + 1e-6)
        corr = float(np.abs((sig * ys).mean()))
        rng = float(np.linalg.norm(col.max(axis=0) - col.min(axis=0)))
        score = corr * rng
        if score > best_score:
            best_score = score
            best_x = x

    if best_x is None:
        return None

    x0, x1 = max(0, best_x - 3), min(W, best_x + 3)
    band = core[:, x0:x1, :]  # (Hc, w, 3)

    # Per-row median excluding near-white and near-black (ticks/labels)
    row = []
    for y in range(Hc):
        px = band[y].reshape(-1, 3)
        b = px.mean(axis=1)
        m = (b < 245) & (b > 20)
        if m.sum() == 0:
            row.append(np.array([255, 255, 255], dtype=np.float32))
        else:
            row.append(np.median(px[m], axis=0))
    row = np.stack(row)

    bright = row.mean(axis=1)
    cf = np.linalg.norm(row - bright[:, None], axis=1)
    valid = np.where((bright < 245) & (cf > np.quantile(cf, 0.4)))[0]
    if valid.size < 10:
        y_start, y_end = int(Hc * 0.10), int(Hc * 0.90)
    else:
        y_start = int(np.quantile(valid, 0.05))
        y_end = int(np.quantile(valid, 0.95))

    # Sample top→bottom
    qs = np.linspace(0.10, 0.90, n)
    ysamp = [int(y_start + float(q) * (y_end - y_start)) for q in qs]
    cols = row[ysamp]
    hexes = ["#%02x%02x%02x" % tuple(np.clip(np.round(c), 0, 255).astype(int)) for c in cols]
    return hexes

# If JSON exists, we will ONLY redraw from it (no simulator imports, no reruns).
_NEED_BENCH = not (LOAD_FROM_JSON and os.path.exists(JSON_PATH))

# ── Fixed shared params (same across all panels) ──────────────────────────────
if LOAD_FROM_JSON and os.path.exists(JSON_PATH):
    with open(JSON_PATH, "r") as f:
        _data = json.load(f)

    _meta = _data.get("meta", {})
    N_COMP = list(_meta.get("N_COMP", N_COMP))
    N_SCALE = list(_meta.get("N_SCALE", N_SCALE))
    N_ALL = sorted(set(N_COMP + N_SCALE))
    N_SEEDS = int(_meta.get("N_SEEDS", N_SEEDS))
    NOISE_HIGH = float(_meta.get("NOISE_HIGH", NOISE_HIGH))
    TOPOLOGY = str(_meta.get("TOPOLOGY", TOPOLOGY))
    WEIGHTS = list(_meta.get("WEIGHTS", WEIGHTS))

    # JSON stores dict keys as strings.
    shared_params = {int(k): v for k, v in _data.get("shared_params", {}).items()}
    _res = _data.get("results", {})
    mean2_12 = {int(k): float(v) for k, v in _res.get("mean2_12", {}).items()}
    mean2_1p = {int(k): float(v) for k, v in _res.get("mean2_1p", {}).items()}
    mean1 = {int(k): float(v) for k, v in _res.get("mean1", {}).items()}
    mae = {int(k): float(v) for k, v in _res.get("mae", {}).items()}
else:
    import time
    from sbi_paulinoise.paulinoise_simulator_2 import NoisyCircuitSimulator as Sim2, get_sbi_observables
    from sbi_paulinoise.paulinoise_simulator_1 import NoisyCircuitSimulator as Sim1

    rng = np.random.default_rng(42)
    shared_params = {}
    for n in N_ALL:
        sim_tmp = Sim2(n_qubits=n, cx_topology=TOPOLOGY)
        shared_params[n] = [rng.uniform(0, NOISE_HIGH, size=sim_tmp.total_params)
                            for _ in range(N_SEEDS)]

# ── sim2 1&2-local benchmark (panels a, b, c) ─────────────────────────────────
if _NEED_BENCH:
    mean2_12 = {}
    print("sim2 (1&2-local):")
    for n in N_ALL:
        sim = Sim2(n_qubits=n, cx_topology=TOPOLOGY)
        obs = get_sbi_observables(n_qubits=n, weights=WEIGHTS)
        times = []
        for p in shared_params[n]:
            t0 = time.perf_counter()
            sim.compute_expectations(p, obs)
            times.append(time.perf_counter() - t0)
        mean2_12[n] = np.mean(times)
        print(f"  n={n:3d}  k={len(obs):4d}  {np.mean(times)*1e3:8.2f} ms")

# ── sim2 single-Pauli benchmark (panel c only) ────────────────────────────────
if _NEED_BENCH:
    mean2_1p = {}
    print("\nsim2 (single Pauli Z_1Z_2):")
    for n in N_SCALE:
        sim = Sim2(n_qubits=n, cx_topology=TOPOLOGY)
        obs_1p = ["ZZ" + "I" * (n - 2)]   # Z on qubits 0,1; I elsewhere
        times = []
        for p in shared_params[n]:
            t0 = time.perf_counter()
            sim.compute_expectations(p, obs_1p)
            times.append(time.perf_counter() - t0)
        mean2_1p[n] = np.mean(times)
        print(f"  n={n:3d}  {np.mean(times)*1e3:8.3f} ms")

# ── sim1 benchmark + MAE (panels a, b) ───────────────────────────────────────
if _NEED_BENCH:
    mean1 = {}
    mae   = {}
    print("\nsim1 (Aer density matrix, 1&2-local):")
    for n in N_COMP:
        sim1 = Sim1(n_qubits=n, cx_topology=TOPOLOGY)
        sim2 = Sim2(n_qubits=n, cx_topology=TOPOLOGY)
        obs  = get_sbi_observables(n_qubits=n, weights=WEIGHTS)
        times, maes = [], []
        for p in shared_params[n]:
            t0 = time.perf_counter()
            x1 = np.array(sim1.compute_expectations(p, obs, method='aer_dm'))
            times.append(time.perf_counter() - t0)
            x2 = np.array(sim2.compute_expectations(p, obs))
            maes.append(np.mean(np.abs(x1 - x2)))
        mean1[n] = np.mean(times)
        mae[n]   = np.mean(maes)
        print(f"  n={n:3d}  {np.mean(times)*1e3:8.2f} ms  MAE={np.mean(maes):.2e}")

# ── Arrays ────────────────────────────────────────────────────────────────────
n_comp  = np.array(N_COMP,  dtype=float)
n_scale = np.array(N_SCALE, dtype=float)

m1_comp   = np.array([mean1[n]    for n in N_COMP])
m2_comp   = np.array([mean2_12[n] for n in N_COMP])
mae_arr   = np.array([mae[n]      for n in N_COMP])
m2_12_sc  = np.array([mean2_12[n] for n in N_SCALE])
m2_1p_sc  = np.array([mean2_1p[n] for n in N_SCALE])

# Reference lines fitted at n=10
ref_n2 = (m2_12_sc[0] / N_SCALE[0] ** 2) * n_scale ** 2
ref_n1 = (m2_1p_sc[0] / N_SCALE[0] ** 1) * n_scale ** 1

# ── Plot style ────────────────────────────────────────────────────────────────
_RC = {
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "DejaVu Sans", "Helvetica"],
    # Bigger fonts for paper/presentation readability
    "font.size":          18,
    "axes.labelsize":     20,
    "axes.titlesize":     20,
    "xtick.labelsize":    18,
    "ytick.labelsize":    18,
    "legend.fontsize":    17,
    "axes.linewidth":     1.2,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.03,
}
_TP  = dict(direction="out", length=5, width=1.2)
_TPm = dict(direction="out", length=3, width=0.9)

# Pull 4 colors from the reference colorbar (top→bottom: high→low).
_REF_PDF = os.path.join(
    _ROOT,
    "sbi_qst_results",
    "qst_prep_sweep_2026-04-16_23-32-29",
    "qst_median_infidelity_landscape.pdf",
)
_pal = _extract_colorbar_palette(_REF_PDF, n=4)
# Fallback to hard-coded extraction from the reference PDF render if auto-extract fails.
_pal = _pal or ["#d38d95", "#f0ddde", "#bbdae9", "#5dabce"]

# Assign colors used across panels.
_C_ORANGE = _pal[0]  # high (reddish)
_C_PURPLE = _pal[1]  # near-white/pink
_C_GREEN  = _pal[2]  # light blue
_C_BLUE   = _pal[3]  # low (blue)

# Thicker lines & larger markers (paper-friendly).
_LW = 2.8
_MS = 7.5
_LW_REF = _LW

with plt.rc_context(_RC):
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 8.4), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.ravel()

    # ── (a) MAE ───────────────────────────────────────────────────────────────
    # Match requested style: panel (a) uses Aer red.
    ax1.plot(n_comp, mae_arr, "o-", color=_C_ORANGE, lw=_LW, ms=_MS)
    ax1.set_xlabel("$n$")
    ax1.set_ylabel("MAE")
    ax1.set_xticks([2, 6, 10, 14])
    ax1.set_yticks([1e-16, 2e-16, 3e-16])
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    _fmt1 = ticker.ScalarFormatter(useMathText=True)
    _fmt1.set_powerlimits((0, 0))
    ax1.yaxis.set_major_formatter(_fmt1)
    ax1.set_ylabel("MAE")
    ax1.tick_params(axis="both", which="major", **_TP)

    # ── (b) Wall time comparison, log y ──────────────────────────────────────
    ax2.plot(n_comp, m1_comp * 1e3, "s-", color=_C_ORANGE, lw=_LW, ms=_MS, label="Aer")
    ax2.plot(n_comp, m2_comp * 1e3, "o-", color=_C_BLUE,   lw=_LW, ms=_MS, label="Ours")
    ax2.set_xlabel("$n$")
    ax2.set_ylabel("Time [ms]")
    ax2.set_yscale("log")
    ax2.set_xticks([2, 6, 10, 14])
    ax2.set_yticks([1, 100, 10000])
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: "10k" if y == 10000 else f"{y:g}"))
    ax2.tick_params(axis="both", which="major", **_TP)
    ax2.tick_params(axis="both", which="minor", **_TPm)
    ax2.legend(frameon=False)

    # ── (c) Single Pauli scaling, log-log ────────────────────────────────────
    # Requested: data lines use red; scaling reference uses blue.
    ax3.plot(n_scale, m2_1p_sc * 1e3, "o-", color=_C_ORANGE, lw=_LW, ms=_MS,
             label=r"$\langle Z_1 Z_2 \rangle$")
    ax3.plot(n_scale, ref_n1   * 1e3, "--", color=_C_BLUE,  lw=_LW_REF, alpha=1.0,
             label=r"$\mathcal{O}(n)$")
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("$n$")
    ax3.set_ylabel("Time [ms]")
    ax3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    # Exactly 4 "nice" y ticks on log scale.
    _y = (m2_1p_sc * 1e3)
    _ymin = float(np.nanmin(_y[_y > 0]))
    _ymax = float(np.nanmax(_y))
    _d0 = int(np.floor(np.log10(_ymin)))
    _d1 = int(np.ceil(np.log10(_ymax)))
    _yticks = np.logspace(_d0, _d1, 4)
    ax3.set_yticks(_yticks)
    ax3.yaxis.set_minor_locator(ticker.NullLocator())
    ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax3.tick_params(axis="both", which="major", **_TP)
    ax3.tick_params(axis="both", which="minor", **_TPm)
    ax3.legend(frameon=False)

    # ── (d) 1&2-local scaling, log-log ───────────────────────────────────────
    ax4.plot(n_scale, m2_12_sc * 1e3, "s-", color=_C_ORANGE, lw=_LW, ms=_MS,
             label="1&2 local")
    ax4.plot(n_scale, ref_n2   * 1e3, "--", color=_C_BLUE,  lw=_LW_REF, alpha=1.0,
             label=r"$\mathcal{O}(n^2)$")
    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.set_xlabel("$n$")
    ax4.set_ylabel("Time [ms]")
    ax4.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax4.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda y, _: "10k" if y == 10000 else ("1k" if y == 1000 else f"{y:g}"))
    )
    ax4.tick_params(axis="both", which="major", **_TP)
    ax4.tick_params(axis="both", which="minor", **_TPm)
    ax4.legend(frameon=False)

    out_path = os.path.join(_HERE, "benchmark_pp_paulinoise.pdf")
    fig.savefig(out_path)
    print(f"\nSaved → {out_path}")

    # Save generated benchmark data alongside the figure for reproducibility.
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

    data_out = {
        "meta": {
            "script": os.path.basename(__file__),
            "N_SEEDS": int(N_SEEDS),
            "NOISE_HIGH": float(NOISE_HIGH),
            "TOPOLOGY": str(TOPOLOGY),
            "WEIGHTS": list(WEIGHTS),
            "N_COMP": list(N_COMP),
            "N_SCALE": list(N_SCALE),
            "rng_seed": 42,
        },
        "shared_params": shared_params,   # dict: n -> list[N_SEEDS] of (total_params,) arrays
        "results": {
            "mean2_12": mean2_12,         # dict: n -> mean wall time [s]
            "mean2_1p": mean2_1p,         # dict: n -> mean wall time [s] (single Pauli)
            "mean1": mean1,               # dict: n -> mean wall time [s] (Aer density matrix)
            "mae": mae,                   # dict: n -> mean absolute error
        },
        "arrays": {
            "n_comp": n_comp,
            "n_scale": n_scale,
            "m1_comp": m1_comp,
            "m2_comp": m2_comp,
            "mae_arr": mae_arr,
            "m2_12_sc": m2_12_sc,
            "m2_1p_sc": m2_1p_sc,
            "ref_n2": ref_n2,
            "ref_n1": ref_n1,
        },
    }
    json_path = os.path.join(_HERE, "benchmark_pp_paulinoise.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(data_out), f, indent=2)
    print(f"Saved → {json_path}")
    if SHOW_FIG:
        plt.show()
