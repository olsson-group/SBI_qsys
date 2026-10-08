"""Evaluate the Trotter NPE posterior against the noisy baseline and ZNE; plot trotter_results.pdf."""

import os
from typing import Dict, List, Optional

import numpy as np
import torch

from sbi_dt.trotterization.evaluation import eval_ideal_noisy_zne, reproduce as _reproduce, save_results
from sbi_dt.trotterization.runner import TrotterNoisyRunner

# Figure style of the NPE results (smaller canvas than the denoiser figure)
STYLE_NPE = dict(
    figsize=(10, 8), fs_label=16, fs_tick=14, fs_xtick_bar=14, fs_ratio=14, lw=2.2,
    tick_kw={}, y_nbins=5, coupling=r"$J$", ratio_split=False,
    legend_fs=14, legend_y=0.04, legend_kw={},
)


def evaluate_and_plot(
    posterior,
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
    n_posterior_samples: int = 200,
) -> None:
    """Compare NPE-denoised, noisy and ZNE expectations on held-out J values and extra depths.

    The denoised value is x_noisy + E[Δx | x_noisy, n_steps], with Δx = x_ideal - x_noisy sampled
    from the posterior. noise_params is the true noise applied by the TrotterNoisyRunner. Writes
    trotter_dynamics_results.json and trotter_results.pdf to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
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

    all_ideal, all_noisy, all_denoised = [], [], []
    all_zne: Dict[str, list] = {m: [] for m in methods}
    for j_i, J in enumerate(J_values):
        print(f"  Evaluating J={J:.3f} ({j_i+1}/{len(J_values)}) ...")
        ideal, noisy, zne = eval_ideal_noisy_zne(
            runner, n_qubits, J, h, dt, n_steps_eval, observables, zne_scale_factors, methods)
        denoised = []
        for i, n_steps in enumerate(n_steps_eval):
            x_obs = torch.cat([
                torch.tensor(noisy[i], dtype=torch.float32, device=device),
                torch.log(torch.tensor([n_steps], dtype=torch.float32, device=device) + 1.0),
            ])
            samples = posterior.sample((n_posterior_samples,), x=x_obs, show_progress_bars=False)
            denoised.append(noisy[i] + samples.mean(dim=0).cpu().numpy())
        all_ideal.append(ideal)
        all_noisy.append(noisy)
        all_denoised.append(np.array(denoised, dtype=np.float32))
        for m in methods:
            all_zne[m].append(zne[m])

    methods = [m for m in methods if all_zne[m]]
    save_results(output_dir, config, J_values, np.array(n_steps_eval, dtype=float) * dt,
                 all_ideal, all_noisy, all_denoised, all_zne, methods, zne_scale_factors, STYLE_NPE)


def reproduce(results_dir: str) -> None:
    """Redraw trotter_results.pdf from the trotter_dynamics_results.json in results_dir."""
    _reproduce(results_dir, STYLE_NPE)
