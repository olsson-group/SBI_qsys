"""Trotter dynamics pipeline: SBI noise -> Clifford training data -> residual denoiser -> evaluation.

Run from the repository root:  python sbi_dt/trotterization/run.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sbi_dt.trotterization import (
    TrotterResidualDenoiser,
    evaluate_and_plot,
    generate_trotter_data,
    get_tfim_observables,
    train_denoiser,
)
from sbi_dt.trotterization.pipeline import load_noise_inputs, load_or_generate_train_data, setup_run

CONFIG = {
    # Paths
    "sbi_params_file": "sbi_paulinoise_results/12q_npe_brickwall/paulinoise_inference_params.csv",
    "true_noise_params_file": "sbi_paulinoise_results/ground_truth/ground_truth_12q/brickwall/paulinoise_1_parameters.csv",
    "output_dir": "sbi_dt_results/trotter",
    "cache_data": True,
    "data_cache_dir": "sbi_dt_results/sbi_dt_data_cache/trotter",

    # TFIM system: H = -J Σ Z_i Z_{i+1} - h Σ X_i
    "n_qubits": 12,
    "cx_topology": "brickwall",   # must match the SBI simulator
    "h": 1.0,
    "dt": 0.1,

    # Trotter depths
    "n_steps_train": list(range(1, 15)),
    "n_steps_eval": list(range(1, 31)),   # extrapolates beyond the training depths

    # Training data: random Clifford circuits, noise = mean of the SBI posterior
    "circuits_per_combo": 500,            # circuits per Trotter depth
    "initial_non_clifford": False,        # True: start from a random Ry·Rx product state

    # Held-out coupling values for evaluation
    "J_eval": np.linspace(-np.pi, 0.0, 50).tolist(),

    # Residual denoiser and training
    "hidden_dim": 756,
    "mlp_num_layers": 2,
    "epochs": 200,
    "lr": 1e-3,
    "batch_size": 256,

    # Evaluation baselines and Aer settings
    "zne_scale_factors": [1, 3, 5],
    "zne_methods": ["linear", "polynomial"],   # subset of linear / polynomial / exponential
    "aer_device": "auto",                      # "auto": GPU when available
    "aer_max_parallel_threads": 0,             # 0: all CPU threads
    "aer_max_parallel_experiments": 0,         # 0: parallelise across the circuits of a batch
    "gpu_batch_size": 64,                      # circuits per Aer GPU job

    "n_workers": -1,   # -1: all CPU cores for data generation
    "random_seed": 42,
}


def main() -> None:
    cfg = dict(CONFIG)
    n_qubits = cfg["n_qubits"]
    run = setup_run(cfg)
    output_dir, device = run["output_dir"], run["device"]

    sbi_samples, true_noise_params = load_noise_inputs(cfg)
    observables = get_tfim_observables(n_qubits)
    n_obs = len(observables)
    print(f"  Observables: {n_obs}")
    print(f"  Train depths: 1..{max(cfg['n_steps_train'])}   Eval depths: 1..{max(cfg['n_steps_eval'])}")

    print("\n=== Phase 1: Generating Trotter training data ===")
    train_data = load_or_generate_train_data(
        cfg, n_qubits, sbi_samples,
        lambda: generate_trotter_data(
            sbi_samples, n_qubits, cfg["n_steps_train"], observables,
            circuits_per_combo=cfg["circuits_per_combo"], random_seed=cfg["random_seed"],
            n_workers=cfg["n_workers"], initial_non_clifford=cfg["initial_non_clifford"],
        ),
    )
    print(f"  Train: {len(train_data['noisy'])} samples")

    print("\n=== Phase 2: Training ===")
    model = TrotterResidualDenoiser(n_obs, cfg["hidden_dim"], cfg["mlp_num_layers"])
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    train_losses = train_denoiser(
        model, train_data, epochs=cfg["epochs"], lr=cfg["lr"], batch_size=cfg["batch_size"], device=device,
    )
    print(f"  Final train loss: {train_losses[-1]:.6f}")
    model_path = os.path.join(output_dir, "trotter_denoiser.pt")
    torch.save({"state_dict": model.state_dict(), "config": cfg, "n_obs": n_obs,
                "noise_dim": sbi_samples.shape[1]}, model_path)
    print(f"  Saved: {model_path}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(train_losses, label="Train loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend(frameon=False)
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_loss.pdf"), bbox_inches="tight")
    plt.close()

    print("\n=== Phase 3: Evaluating ===")
    evaluate_and_plot(
        model, true_noise_params, n_qubits, cfg["h"], cfg["dt"],
        cfg["n_steps_train"], cfg["n_steps_eval"], observables,
        J_eval=cfg["J_eval"], output_dir=output_dir, random_seed=cfg["random_seed"], device=device,
        cx_topology=cfg["cx_topology"], config=cfg,
        zne_scale_factors=cfg["zne_scale_factors"], zne_methods=cfg["zne_methods"],
        aer_device=cfg["aer_device"],
        aer_max_parallel_threads=cfg["aer_max_parallel_threads"],
        aer_max_parallel_experiments=cfg["aer_max_parallel_experiments"],
        gpu_batch_size=cfg["gpu_batch_size"],
    )
    print(f"\nAll outputs in {output_dir}/")


if __name__ == "__main__":
    main()
