"""Zero-noise extrapolation by probabilistic error amplification (PEA) with SBI-inferred noise.

Edit CONFIG below, then run from the repository root:
    python sbi_pec_zne/zne/run.py
"""

import json
import os
import random
import sys
import zlib
from collections import Counter

import numpy as np

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sbi_pec_zne.common import (
    aer_reference_expectations,
    build_aer_noise_model,
    build_circuit,
    load_sbi_params,
    parity_expectation,
)
from sbi_pec_zne.gpu_utils import make_aer_sim
from sbi_pec_zne.zne.extrapolate import extrapolate_to_zero
from sbi_pec_zne.zne.sampling import amplify_rates, sample_pauli_configs_batch
from sbi_pec_zne.zne.visualize import load_and_plot as _load_and_plot

CONFIG = {
    # Inputs
    "sbi_results_dir": "sbi_paulinoise_results/15q_npe_brickwall",
    "ground_truth_dir": "sbi_paulinoise_results/ground_truth/ground_truth_15q/brickwall",
    "n_qubits": 15,
    "cx_topology": "brickwall",
    "target_observables": ["IIIIZZZZIIIIIII"],

    # ZNE: noise gains, circuit instances sampled per gain, and shots per instance
    "gains": [1, 1.6, 2.2, 2.8],
    "n_circuit_instances": 2000,
    "shots_per_instance": 256,
    "run_param_summaries": ["mean", "median", "truth"],
    "extrapolation_methods": ["linear", "quadratic", "exponential"],

    # "aer_dm": Aer density matrix with the true noise.
    # "paulinoise2": paulinoise_simulator_2 (Heisenberg back-propagation) + binomial shot noise, no Aer.
    "SIMULATION_METHOD": "aer_dm",

    "output_dir": "sbi_pec_zne_results/zne",
    "random_seed": 42,
}


def run_zne_experiment(config: dict) -> dict:
    """Probabilistic error amplification ZNE.

    The device has the true noise Λ. For a gain G, Paulis sampled from the amplified channel Λ^(G-1) are
    inserted after every CNOT so that the total noise is Λ^G (no insertions at G = 1). ⟨O⟩ is measured for
    every gain and extrapolated to G = 0, once per parameter summary ("mean", "median", "truth") of the
    SBI posterior that defines the amplified channel.

    Writes zne_results.json to ``output_dir`` and returns {summary: {observable: result}}.
    """
    n_qubits = config["n_qubits"]
    n_cnots = n_qubits - 1
    n_gate = n_cnots * 15
    cx_topology = config.get("cx_topology", "brickwall")
    observables = config["target_observables"]
    gains = sorted(config["gains"])
    n_instances = config["n_circuit_instances"]
    shots_per_instance = config["shots_per_instance"]
    extrap_methods = config["extrapolation_methods"]
    output_dir = config["output_dir"]
    seed = config.get("random_seed", 123)
    method = config.get("SIMULATION_METHOD", "aer_dm")
    if method not in ("aer_dm", "paulinoise2"):
        raise ValueError("config['SIMULATION_METHOD'] must be 'aer_dm' or 'paulinoise2' for ZNE.")

    params = load_sbi_params(config["sbi_results_dir"], config["ground_truth_dir"])
    summaries = [k for k in config.get("run_param_summaries") or ("mean", "median", "truth") if k in params]
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)
    random.seed(seed)

    truth_rates = params["truth"][:n_gate]   # the device noise is the true noise
    if method == "paulinoise2":
        from sbi_pec_zne.paulinoise2_backend import (
            compute_ideal_paulinoise2,
            compute_noisy_paulinoise2,
            make_paulinoise2_sim,
            paulinoise2_measure_parity,
        )
        sim2 = make_paulinoise2_sim(n_qubits, cx_topology=cx_topology)
        ideal_expectations = compute_ideal_paulinoise2(sim2, observables)
        noisy_expectations = compute_noisy_paulinoise2(sim2, truth_rates, observables)
        print("[ZNE] SIMULATION_METHOD=paulinoise2 — paulinoise_simulator_2 + binomial shot noise")
    else:
        noise_model = build_aer_noise_model(truth_rates, n_qubits)
        aer_sim = make_aer_sim(method="density_matrix", noise_model=noise_model, seed=seed)
        ideal_expectations, noisy_expectations = aer_reference_expectations(
            n_qubits, observables, cx_topology, noise_model, seed)
        print("[ZNE] SIMULATION_METHOD=aer_dm — Aer density_matrix with truth noise")
    print(f"[ZNE] Ideal ⟨O⟩: {ideal_expectations}")
    print(f"[ZNE] Noisy ⟨O⟩ (unmitigated): {noisy_expectations}")

    all_results = {}
    expectations_by_gain = {}
    for name in summaries:
        rates_matrix = params[name][:n_gate].reshape(n_cnots, 15)
        all_results[name] = {}
        expectations_by_gain[name] = {}

        for obs_str in observables:
            ev_at_gain = {}
            for G in gains:
                # Insert Λ^(G-1) on top of the device's Λ; nothing at G = 1.
                rates_ins = amplify_rates(rates_matrix, G - 1.0) if G > 1.0 else np.zeros_like(rates_matrix)
                rng = np.random.default_rng(seed + zlib.crc32(f"{name}|{obs_str}|{G}".encode()) % (2**31))
                configs = sample_pauli_configs_batch(rates_ins, n_instances, rng)
                counter = Counter(configs)
                print(f"[ZNE] G={G} ({name}): {len(counter)} unique configs, "
                      f"per_instance_shots={shots_per_instance} → {n_instances * shots_per_instance} total shots "
                      f"({n_instances}×{shots_per_instance})")

                weighted_sum = 0.0
                for i, (key, count) in enumerate(counter.items()):
                    n_shots = count * shots_per_instance
                    if method == "paulinoise2":
                        meas_rng = np.random.default_rng(seed + i + int(G * 1000))
                        ev = paulinoise2_measure_parity(key, truth_rates, obs_str, n_shots, sim2, meas_rng)
                    else:
                        qc = build_circuit(key, n_qubits, obs_str, cx_topology)
                        job = aer_sim.run(qc, shots=n_shots, seed_simulator=seed + i + int(G * 1000))
                        ev = parity_expectation(job.result().get_counts(), obs_str)
                    weighted_sum += count * ev
                ev_at_gain[G] = float(weighted_sum / n_instances)

            expectations_by_gain[name][obs_str] = ev_at_gain
            ev_arr = np.array([ev_at_gain[g] for g in gains])
            extrapolations = {}
            for m in extrap_methods:
                val, info = extrapolate_to_zero(np.array(gains), ev_arr, method=m)
                extrapolations[m] = {"value": val, "fit_info": info}
            first = extrapolations[extrap_methods[0]]
            all_results[name][obs_str] = {
                "extrapolated": first["value"],
                "ev_by_gain": ev_at_gain,
                "fit_info": first["fit_info"],
                "extrapolations": extrapolations,
            }
            print(f"[ZNE] {name} / {obs_str}: " + ", ".join(
                f"{m}→{extrapolations[m]['value']:.4f}" for m in extrap_methods))

    out_json = os.path.join(output_dir, "zne_results.json")
    with open(out_json, "w") as f:
        json.dump({
            "gains": gains,
            "observables": observables,
            "results": all_results,
            "expectations_by_gain": expectations_by_gain,
            "ideal_expectations": ideal_expectations,
            "noisy_expectations": noisy_expectations,
            "extrapolation_methods": extrap_methods,
            "n_qubits": n_qubits,
            "cx_topology": cx_topology,
            "n_circuit_instances": n_instances,
            "shots_per_instance": shots_per_instance,
        }, f, indent=2)
    print(f"[ZNE] Results saved → {out_json}")
    return all_results


def _main():
    np.random.seed(CONFIG["random_seed"])
    random.seed(CONFIG["random_seed"])
    print("=== PEA-ZNE using SBI-inferred noise parameters ===")
    print(f"SBI results: {CONFIG['sbi_results_dir']}")
    print(f"Observables: {CONFIG['target_observables']}")
    print(f"Gains: {CONFIG['gains']}")
    print(f"Output dir: {CONFIG['output_dir']}\n")
    run_zne_experiment(CONFIG)
    res_json = os.path.join(CONFIG["output_dir"], "zne_results.json")
    if os.path.isfile(res_json):
        _load_and_plot(res_json, CONFIG["output_dir"])


if __name__ == "__main__":
    _main()
