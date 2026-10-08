"""Trotter NPE pipeline: SBI noise -> Clifford training data -> NSF posterior over x_ideal - x_noisy -> evaluation.

Run from the repository root:  python sbi_dt/trotterization_sbi/run.py
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

from sbi_dt.trotterization import generate_trotter_data, get_tfim_observables
from sbi_dt.trotterization.pipeline import load_noise_inputs, load_or_generate_train_data, setup_run
from sbi_dt.trotterization_sbi import evaluate_and_plot, train_npe

CONFIG = {
    # Paths
    "sbi_params_file": "sbi_paulinoise_results/12q_npe_brickwall/paulinoise_inference_params.csv",
    "true_noise_params_file": "sbi_paulinoise_results/ground_truth/ground_truth_12q/brickwall/paulinoise_1_parameters.csv",
    "output_dir": "sbi_dt_results/trotter_npe",
    "cache_data": True,
    "data_cache_dir": "sbi_dt_results/sbi_dt_data_cache/trotter_npe",

    # TFIM system: H = -J Σ Z_i Z_{i+1} - h Σ X_i
    "n_qubits": 12,
    "cx_topology": "brickwall",   # must match the SBI simulator
    "h": 1.0,
    "dt": 0.1,

    # Trotter depths
    "n_steps_train": list(range(1, 15)),
    "n_steps_eval": list(range(1, 31)),   # extrapolates beyond the training depths

    # Training data: random Clifford circuits from a random Ry·Rx product state (x_ideal is continuous),
    # noise = mean of the SBI posterior
    "circuits_per_combo": 500,            # circuits per Trotter depth
    "initial_non_clifford": True,

    # Held-out coupling values for evaluation
    "J_eval": np.linspace(-np.pi, 0.0, 50).tolist(),

    # zuko NSF posterior
    "flow_hidden_features": 2048,
    "num_transforms": 3,
    "nsf_num_bins": 16,

    # Training
    "epochs": 5000,
    "stop_after_epochs": 100,    # early-stopping patience
    "lr": 1e-4,
    "batch_size": 500,
    "clip_max_norm": 5.0,

    # Evaluation
    "n_posterior_samples": 1,    # the posterior is tight, so one sample is close to its mean
    "zne_scale_factors": [1, 3, 5],
    "zne_methods": ["linear", "polynomial"],
    "aer_device": "auto",
    "aer_max_parallel_threads": 0,
    "aer_max_parallel_experiments": 0,
    "gpu_batch_size": 128,

    "n_workers": -1,
    "random_seed": 42,
}


def main() -> None:
    cfg = dict(CONFIG)
    n_qubits = cfg["n_qubits"]
    run = setup_run(cfg)
    output_dir, device = run["output_dir"], run["device"]

    sbi_samples, true_noise_params = load_noise_inputs(cfg)
    observables = get_tfim_observables(n_qubits)
    print(f"  Observables: {len(observables)}")
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

    print("\n=== Phase 2: Training NSF posterior (sbi NPE) ===")
    print(f"  hidden_features={cfg['flow_hidden_features']}, num_transforms={cfg['num_transforms']}, "
          f"num_bins={cfg['nsf_num_bins']}")
    posterior, estimator, train_log_probs, val_log_probs = train_npe(train_data, cfg, device)
    estimator_path = os.path.join(output_dir, "trotter_npe_estimator.pt")
    torch.save(estimator, estimator_path)
    print(f"  Saved estimator: {estimator_path}")

    if train_log_probs:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(train_log_probs, label="Train log-prob")
        if val_log_probs:
            ax.plot(val_log_probs, label="Val log-prob")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Log-prob")
        ax.legend(frameon=False)
        ax.grid(False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_loss.pdf"), bbox_inches="tight")
        plt.close()
        print(f"  Final train log-prob: {train_log_probs[-1]:.4f}")

    print("\n=== Phase 3: Evaluating ===")
    evaluate_and_plot(
        posterior, true_noise_params, n_qubits, cfg["h"], cfg["dt"],
        cfg["n_steps_train"], cfg["n_steps_eval"], observables,
        J_eval=cfg["J_eval"], output_dir=output_dir, random_seed=cfg["random_seed"], device=device,
        cx_topology=cfg["cx_topology"], config=cfg,
        zne_scale_factors=cfg["zne_scale_factors"], zne_methods=cfg["zne_methods"],
        aer_device=cfg["aer_device"],
        aer_max_parallel_threads=cfg["aer_max_parallel_threads"],
        aer_max_parallel_experiments=cfg["aer_max_parallel_experiments"],
        gpu_batch_size=cfg["gpu_batch_size"],
        n_posterior_samples=cfg["n_posterior_samples"],
    )
    print(f"\nAll outputs in {output_dir}/")


if __name__ == "__main__":
    main()
