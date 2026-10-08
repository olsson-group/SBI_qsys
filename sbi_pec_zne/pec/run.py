"""Probabilistic error cancellation (PEC) with SBI-inferred noise.

Edit CONFIG below, then run from the repository root:
    python sbi_pec_zne/pec/run.py
"""

import os
import random
import sys

import numpy as np

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sbi_pec_zne.pec.correct import run_pec_experiment
from sbi_pec_zne.pec.visualize import load_and_plot

CONFIG = {
    # Inputs: SBI posterior samples and the true noise parameters
    "sbi_results_dir": "sbi_paulinoise_results/15q_npe_brickwall",
    "ground_truth_dir": "sbi_paulinoise_results/ground_truth/ground_truth_15q/brickwall",

    # System
    "n_qubits": 15,
    "cx_topology": "brickwall",   # brickwall | linear

    # PEC: observable, sample-size checkpoints, and which parameter summaries build the quasi-probabilities
    "target_observables": ["IIIIZZZZIIIIIII"],
    "sample_sizes": [1000, 2000, 3000, 4000, 5000],
    "run_param_summaries": ["mean", "median", "truth"],   # subset of mean / median / truth

    # The device always runs with the true noise.
    # "aer_dm": Aer density matrix. "paulinoise2": paulinoise_simulator_2 + binomial shot noise, no Aer.
    "SIMULATION_METHOD": "aer_dm",

    # A circuit drawn c times gets c × per_instance_shots shots
    "per_instance_shots": 256,

    "output_dir": "sbi_pec_zne_results/pec",
    "random_seed": 42,
}


def _main():
    np.random.seed(CONFIG["random_seed"])
    random.seed(CONFIG["random_seed"])
    print("=== PEC using SBI-inferred noise parameters ===")
    print(f"SBI results: {CONFIG['sbi_results_dir']}")
    print(f"Observables: {CONFIG['target_observables']}")
    print(f"SIMULATION_METHOD: {CONFIG['SIMULATION_METHOD']}")
    print(f"Output dir:  {CONFIG['output_dir']}\n")
    run_pec_experiment(CONFIG)
    res_json = os.path.join(CONFIG["output_dir"], "pec_results.json")
    if os.path.isfile(res_json):
        load_and_plot(res_json, CONFIG["output_dir"])


if __name__ == "__main__":
    _main()
