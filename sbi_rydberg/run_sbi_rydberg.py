"""SBI of atom-position displacements on a Rydberg lattice (Pauli-propagation simulator + NPE).

Pipeline: θ (per-atom Δx, Δy) -> Pauli propagation -> <Z_i>, <Z_i Z_j> -> NPE (zuko NSF flow).

Run from the repository root::

    python sbi_rydberg/run_sbi_rydberg.py

Environment variables: SBI_OUTPUT_ROOT (default ``sbi_rydberg_results``),
SBI_REGRESSION_N (default 100), SBI_REGRESSION_POST_N (default 500).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sbi_rydberg.lattice_simulate import RydbergSimConfig
from sbi_rydberg.pauli_propagation_rydberg import simulate_batch_pauli_propagation
from sbi_rydberg.train_npe import train_npe_amortised
from sbi_rydberg.visualize_lattice import (
    plot_all_lattice,
    plot_regression_pooled_rmse,
    plot_regression_scatter,
)

CONFIG = {
    # Lattice and dynamics
    "NX": 9,
    "NY": 9,
    "LATTICE_CONSTANT_UM": 8.0,
    "OMEGA_RAD_PER_US": 15.0,
    "DETUNING_RAD_PER_US": 15.0,
    "TIMES_US": (0.01,),
    "DT_US": 0.01,
    "INTERACTION_SCALE": 1.0,
    "V_THRESHOLD": 1.0,
    "D_MAX": 0.15,  # displacement prior: uniform ±D_MAX [μm]
    # Pauli propagation
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "PP_EPS": 1e-6,
    "PP_MAX_WEIGHT": 5,
    "PP_CHUNK_SIZE": 8192,
    # NPE training
    "NPE_N_SIMS": 100_000,
    "OBS_SEED": 7,
    "EPOCHS": 5000,
    "BATCH_SIZE": 4096,
    "STOP_AFTER_EPOCHS": 100,
    "LR": 1e-4,
    "CLIP_MAX_NORM": 5.0,
    "NUM_POSTERIOR_SAMPLES": 2000,
    # Flow
    "FLOW_MODEL": "zuko_nsf",
    "FLOW_HIDDEN_FEATURES": 2048,
    "NUM_TRANSFORMS": 3,
    "NSF_NUM_BINS": 16,
    "EMBED_DIM": 256,
}


def evaluate_regression(
    posterior, simulate_batch, results_dir: str, *, d_max: float, n_params: int, device: str,
    n_targets: int = 100, n_post_samples: int = 500, seed: int = 123,
) -> None:
    """Posterior accuracy on held-out θ drawn from the prior: testdata, samples, scatter, pooled RMSE."""
    rng = np.random.default_rng(seed)
    theta_true = rng.uniform(-d_max, d_max, size=(n_targets, n_params)).astype(np.float32)
    print(f"\nRegression evaluation: {n_targets} targets × {n_params} params...", flush=True)
    x_all = simulate_batch(theta_true.astype(np.float64))

    samples = np.zeros((n_targets, n_post_samples, n_params), dtype=np.float32)
    for i in range(n_targets):
        x_cond = torch.as_tensor(x_all[i], dtype=torch.float32, device=device).reshape(1, -1)
        s = posterior.sample((n_post_samples,), x=x_cond, show_progress_bars=False)
        samples[i] = s.detach().cpu().numpy()
        if (i + 1) % 10 == 0:
            print(f"  posterior sampling {i + 1}/{n_targets}", flush=True)

    np.savez(
        os.path.join(results_dir, "testdata.npz"),
        theta_true=theta_true, theta_hat=samples.mean(axis=1), theta_std=samples.std(axis=1),
    )
    np.save(os.path.join(results_dir, "regression_theta_samples.npy"), samples)

    scatter_path, r2 = plot_regression_scatter(results_dir, theta_true, samples, d_max)
    rmse_path = plot_regression_pooled_rmse(results_dir, theta_true, samples, d_max)
    with open(os.path.join(results_dir, "figures_index.txt"), "a", encoding="utf-8") as f:
        f.write("\n" + rmse_path)
    print(f"Regression R² = {r2:.4f}   saved → {scatter_path}, {rmse_path}", flush=True)


def main(overrides: dict | None = None) -> str:
    """Run the full pipeline; ``overrides`` replaces CONFIG entries. Returns the results directory."""
    cfg_dict = {**CONFIG, **(overrides or {})}
    torch.manual_seed(0)
    np.random.seed(0)

    nx, ny = int(cfg_dict["NX"]), int(cfg_dict["NY"])
    n_params = 2 * nx * ny
    d_max = float(cfg_dict["D_MAX"])
    device = cfg_dict["DEVICE"]

    cfg = RydbergSimConfig(
        nx=nx,
        ny=ny,
        lattice_constant_um=float(cfg_dict["LATTICE_CONSTANT_UM"]),
        omega_rad_per_us=float(cfg_dict["OMEGA_RAD_PER_US"]),
        detuning_rad_per_us=float(cfg_dict["DETUNING_RAD_PER_US"]),
        times_us=tuple(cfg_dict["TIMES_US"]),
        dt_us=float(cfg_dict["DT_US"]),
        interaction_scale=float(cfg_dict["INTERACTION_SCALE"]),
        v_threshold=float(cfg_dict["V_THRESHOLD"]),
    )
    print(f"Pauli propagation  n_obs={cfg.n_obs}  max_weight={cfg_dict['PP_MAX_WEIGHT']}  "
          f"eps={cfg_dict['PP_EPS']}  device={device}", flush=True)

    def simulate_batch(theta: np.ndarray, verbose: bool = False) -> np.ndarray:
        return simulate_batch_pauli_propagation(
            theta, cfg, eps=float(cfg_dict["PP_EPS"]), max_weight=int(cfg_dict["PP_MAX_WEIGHT"]),
            device=device, chunk_size=int(cfg_dict["PP_CHUNK_SIZE"]), verbose=verbose,
        )

    run_name = datetime.now().strftime(f"rydberg_lattice{nx}x{ny}_tn_%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join(os.environ.get("SBI_OUTPUT_ROOT", "sbi_rydberg_results"), run_name)
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"config": cfg_dict, "nx": nx, "ny": ny, "n_obs": cfg.n_obs}, f, indent=2)

    # Observation: a random θ_true from the prior.
    rng = np.random.default_rng(int(cfg_dict["OBS_SEED"]))
    theta_true = rng.uniform(-d_max, d_max, size=(n_params,)).astype(np.float32)
    np.savetxt(os.path.join(results_dir, "theta_true.csv"), theta_true.reshape(1, -1), delimiter=",")
    print("Simulating x_obs...", flush=True)
    x_obs_np = simulate_batch(theta_true.reshape(1, -1).astype(np.float64), verbose=True)[0]
    np.savetxt(os.path.join(results_dir, "x_obs.csv"), x_obs_np.reshape(1, -1), delimiter=",")

    posterior, samples = train_npe_amortised(
        cfg_dict,
        torch.as_tensor(x_obs_np, dtype=torch.float32, device=device),
        results_dir,
        num_params=n_params,
        prior_low=-d_max,
        prior_high=d_max,
        simulate_batch=simulate_batch,
    )
    np.savetxt(os.path.join(results_dir, "posterior_samples.csv"), samples, delimiter=",")
    plot_all_lattice(results_dir, theta_true=theta_true, posterior_samples=samples, nx=nx, ny=ny)

    err = np.linalg.norm(samples - theta_true.reshape(1, -1), axis=1)
    print(f"\nPosterior ||θ - θ_true||_2 (μm): median = {np.median(err):.4f}  mean = {err.mean():.4f}")

    evaluate_regression(
        posterior, simulate_batch, results_dir, d_max=d_max, n_params=n_params, device=device,
        n_targets=int(os.environ.get("SBI_REGRESSION_N", "100")),
        n_post_samples=int(os.environ.get("SBI_REGRESSION_POST_N", "500")),
    )
    print(f"\nDone. Results in {results_dir}/", flush=True)
    return results_dir


if __name__ == "__main__":
    main()
