"""SBI pipeline entry point + configuration.

Edit CONFIG below, then run:
    python sbi_paulinoise/run_sbi.py
or:
    python -m sbi_paulinoise.run_sbi
"""

import sys
import os as _os

# Allow direct execution from anywhere
if __name__ == "__main__":
    sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import json
import multiprocessing
import numpy as np
import torch
from datetime import datetime


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  — edit here
# ══════════════════════════════════════════════════════════════════════════════
device = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {
    # ── Data files ────────────────────────────────────────────────────────────
      "OBSERVABLES_FILE": "sbi_paulinoise_results/ground_truth/ground_truth_10q/brickwall/paulinoise_observables.csv",                                                                   
      "X_OBS_FILE":       "sbi_paulinoise_results/ground_truth/ground_truth_10q/brickwall/paulinoise_1_observations.csv",                                                                
      "THETA_TRUTH_FILE": "sbi_paulinoise_results/ground_truth/ground_truth_10q/brickwall/paulinoise_1_parameters.csv",
      "INIT_STATES_FILE": "sbi_paulinoise_results/ground_truth/ground_truth_10q/brickwall/paulinoise_init_states.csv",

    # ── Circuit / Simulator ───────────────────────────────────────────────────
    "N_QUBITS":      10,
    "CX_TOPOLOGY":   "brickwall",        # "linear" or "brickwall"
    "PRIOR_HIGH":         0.001,  # BoxUniform upper bound; θ ∈ [0, PRIOR_HIGH]
    "INFER_UPPER_FACTOR": 1.0,  # inference samples allowed up to PRIOR_HIGH * factor (proposal stays strict)
 
    # ── Training ──────────────────────────────────────────────────────────────
    "DEVICE":            device,
    "EPOCHS":            4000,
    "BATCH_SIZE":        2048,
    "STOP_AFTER_EPOCHS": 50,
    "LR":                1e-4,
    "CLIP_MAX_NORM":     5.0,
    "WEIGHT_DECAY":      0,

    # ── Posterior sampling ────────────────────────────────────────────────────
    "NUM_POSTERIOR_SAMPLES": 2000,

    # ── Flow model ────────────────────────────────────────────────────────────
    # FLOW_MODEL: "zuko_nsf" | "zuko_maf" | "zuko_gf" | "mdn"
    "FLOW_MODEL":              "zuko_nsf",
    "FLOW_HIDDEN_FEATURES":    2048,
    "NUM_TRANSFORMS":          3,
    "NSF_NUM_BINS":            16,   

    # ── SNPE ──────────────────────────────────────────────────────────────────
    "SNPE_NUM_ROUNDS":     6,
    # Simulator runs per SNPE round; each returns one full obs vector (init states × Paulis).
    # First round can be overridden by SNPE_SIMS_FIRST_ROUND.
    "SNPE_SIMS_PER_ROUND": 5_000,
    "SNPE_SIMS_FIRST_ROUND": 5_000,
    "SNPE_REFRESH_MODEL":  False,    # True = fresh NPE each round  
    "SNPE_ROUNDS_DIR":     None,    # None → auto time-stamped folder
    "SNPE_SIM_N_JOBS":     16,      # CPU workers for parallel simulation
    "STANDARDIZE_X":       True,    # z-score observations before training
}
# ══════════════════════════════════════════════════════════════════════════════

from sbi_paulinoise import data as _data
from sbi_paulinoise.visualize import reproduce


def _main():
    """Run the full SBI pipeline. Only executed in the main process."""
    # ── Reproducibility ───────────────────────────────────────────────────────
    np.random.seed(123)
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)

    # ── Device info ───────────────────────────────────────────────────────────
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}, "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("No GPU — using CPU")


    # ── Results folder ────────────────────────────────────────────────────────
    _parent = "sbi_paulinoise_results"
    results_dir = os.path.join(_parent, datetime.now().strftime("sbi_results_%Y-%m-%d_%H-%M-%S"))
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "config_snapshot.txt"), "w") as _f:
        _f.write(f"SBI run at {datetime.now().isoformat()}\n")
        for k, v in CONFIG.items():
            _f.write(f"{k}: {v}\n")
    print(f"Results folder: {results_dir}")

    # ── Ground-truth flag: only enable when paulinoise_simulator_1.py uses 'aer_dm' ─────
    plot_ground_truth = False
    _paulinoise1 = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "paulinoise_simulator_1.py")
    if os.path.exists(_paulinoise1):
        try:
            with open(_paulinoise1) as _f:
                _content = _f.read()
            if "SIMULATION_METHOD = 'aer_dm'" in _content:
                plot_ground_truth = True
                print(f"[Detection] aer_dm found in {_paulinoise1} → ground-truth ENABLED.")
            else:
                print(f"[Detection] aer_dm not found in {_paulinoise1} → ground-truth DISABLED.")
        except Exception as _e:
            print(f"[Warning] Could not read {_paulinoise1}: {_e}")
    else:
        print(f"[Warning] {_paulinoise1} not found → ground-truth DISABLED.")

    # ── Load / generate data ──────────────────────────────────────────────────
    print("\nLoading data...")
    info = _data.load_snpe(CONFIG)

    num_params = info["num_params"]
    n_obs = info["n_obs"]
    theta_truth_np = info.get("theta_truth_np")
    plot_gt = plot_ground_truth and (theta_truth_np is not None)

    x_obs_np = info["x_obs_np"]

    # ── Save training dataset metadata to JSON ────────────────────────────────
    _STATE_NAMES = ["++++", "+--", "-+-", "--+"]  # tetrahedral states (sign of X,Y,Z)
    obs_k    = CONFIG.get("OBS_SUBSET_K")
    obs_list = info["observables"]
    n_qubits = CONFIG["N_QUBITS"]
    meta: dict = {
        "n_qubits":        n_qubits,
        "cx_topology":     CONFIG["CX_TOPOLOGY"],
        "learning_mode":   CONFIG.get("LEARNING_MODE", "obs_learning"),
        "obs_subset_k":    obs_k,
        "obs_subset_seed": CONFIG.get("OBS_SUBSET_SEED"),
        "observables":     obs_list,
        "canonical_states": _STATE_NAMES,
    }
    if obs_k is not None and CONFIG.get("LEARNING_MODE", "obs_learning") == "obs_learning":
        n_states_full = 4 ** n_qubits
        n_obs_total   = len(obs_list)
        flat = x_obs_np.flatten()          # shape (obs_k * 3,)
        slots = []
        for i in range(obs_k):
            s_norm = float(flat[i * 3])
            p_norm = float(flat[i * 3 + 1])
            val    = float(flat[i * 3 + 2])
            s_idx  = int(round(s_norm * (n_states_full - 1)))
            p_idx  = int(round(p_norm * (n_obs_total   - 1))) if n_obs_total > 1 else 0
            # base-4 decode: MSB first
            qubit_states = [(s_idx // (4 ** q)) % 4 for q in range(n_qubits - 1, -1, -1)]
            slots.append({
                "slot":             i,
                "state_idx":        s_idx,
                "qubit_states":     [_STATE_NAMES[q] for q in qubit_states],
                "pauli_idx":        p_idx,
                "pauli":            obs_list[p_idx],
                "expectation_value": val,
            })
        meta["x_obs_sampled_slots"] = slots
    else:
        meta["x_obs_note"] = "full mode: all init_states × observables used"

    _meta_path = os.path.join(results_dir, "training_data_meta.json")
    with open(_meta_path, "w") as _f:
        json.dump(meta, _f, indent=2, ensure_ascii=False)
    print(f"  Training data metadata → {_meta_path}")

    x_obs = torch.as_tensor(x_obs_np, dtype=torch.float32).to(device)
    theta_truth = (
        torch.as_tensor(theta_truth_np, dtype=torch.float32).to(device)
        if theta_truth_np is not None else None
    )

    if not plot_gt and theta_truth_np is not None:
        print("  paulinoise_1_parameters.csv found but ground-truth plotting is disabled (simulator not aer_dm).")
    print(f"  num_params={num_params}, n_obs={n_obs}, plot_ground_truth={plot_gt}")

    # ── Train ─────────────────────────────────────────────────────────────────
    from sbi_paulinoise.train_snpe import train_snpe
    posterior, posterior_samples_np, nmae_mean, nmae_median = train_snpe(
        config=CONFIG,
        x_obs=x_obs,
        theta_truth=theta_truth,
        num_params=num_params,
        n_obs=n_obs,
        observables=info["observables"],
        results_dir=results_dir,
        init_prep_blochs=info["init_prep_blochs"],
    )

    # ── Save posterior samples ────────────────────────────────────────────────
    csv_path = os.path.join(results_dir, "paulinoise_inference_params.csv")
    np.savetxt(csv_path, posterior_samples_np, delimiter=",")
    print(f"Saved inference parameters to {csv_path} (shape: {posterior_samples_np.shape})")

    # ── Visualise (use reproduce for consistency) ─────────────────────────────
    print("\nGenerating visualisation (via reproduce)...")
    # `reproduce()` loads saved samples/truth from disk, so the resulting plots
    # match the standalone reproduction workflow.
    reproduce(results_dir)

    print(f"\nDone. All results in {results_dir}/")


if __name__ == "__main__":
    # Only run the pipeline in the main process. When using multiprocessing
    # (e.g. SNPE simulator with ProcessPoolExecutor), spawned workers re-execute
    # this module; we must not run the pipeline again in those workers.
    if multiprocessing.current_process().name == "MainProcess":
        _main()
