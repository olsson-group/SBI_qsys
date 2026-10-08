"""Probabilistic error cancellation (PEC) of the characterisation circuit with SBI-inferred noise."""

import json
import os
from collections import Counter

import numpy as np

from sbi_pec_zne.common import (
    aer_reference_expectations,
    build_aer_noise_model,
    build_circuit,
    load_sbi_params,
    parity_expectation,
)
from sbi_pec_zne.gpu_utils import make_aer_sim
from .sampling import run_pec_sampling


def run_pec_experiment(config: dict) -> dict:
    """Run PEC with the quasi-probabilities built from each SBI parameter summary.

    The device noise is always the true parameters. For every summary in ``run_param_summaries``
    ("mean", "median", "truth") and every observable, quasi-probability samples are drawn once up to the
    largest checkpoint; each checkpoint estimate is γ · mean(sign · ⟨O⟩) over its first N samples.
    A circuit drawn c times is measured with c × ``per_instance_shots`` shots.

    ``SIMULATION_METHOD``: "aer_dm" (Aer density matrix) or "paulinoise2" (paulinoise_simulator_2 with
    binomial shot noise, no Aer). See ``pec/run.py`` for all settings.

    Returns {summary: {observable: [mitigated ⟨O⟩ per checkpoint]}} and writes pec_results.json and
    one pec_sampling_<summary>_<observable>.json per run to ``output_dir``.
    """
    n_qubits = config["n_qubits"]
    n_gate = (n_qubits - 1) * 15
    cx_topology = config.get("cx_topology", "brickwall")
    seed = config.get("random_seed", 123)
    sample_sizes = config["sample_sizes"]
    observables = config["target_observables"]
    output_dir = config["output_dir"]
    per_instance_shots = int(config["per_instance_shots"])
    if per_instance_shots < 1:
        raise ValueError("per_instance_shots must be >= 1")
    method = config.get("SIMULATION_METHOD")
    if method not in ("aer_dm", "paulinoise2"):
        raise ValueError("config['SIMULATION_METHOD'] must be 'aer_dm' or 'paulinoise2'.")
    print(f"[PEC] per_instance_shots={per_instance_shots} "
          f"({len(sample_sizes)} checkpoints; unique circuit k gets count_k×{per_instance_shots} shots)")
    os.makedirs(output_dir, exist_ok=True)

    params = load_sbi_params(config["sbi_results_dir"], config["ground_truth_dir"])
    summaries = [k for k in config.get("run_param_summaries") or ("mean", "median", "truth") if k in params]
    if not summaries:
        raise ValueError(f"run_param_summaries matches none of {list(params)}.")
    print(f"[PEC] Parameter summaries to run: {summaries}")

    device_rates = params["truth"][:n_gate]   # the device noise is the true noise
    if method == "paulinoise2":
        from sbi_pec_zne.paulinoise2_backend import (
            compute_ideal_paulinoise2,
            compute_noisy_paulinoise2,
            make_paulinoise2_sim,
            paulinoise2_measure_parity,
        )
        sim2 = make_paulinoise2_sim(n_qubits, cx_topology=cx_topology)
        ideal_expectations = compute_ideal_paulinoise2(sim2, observables)
        noisy_expectations = compute_noisy_paulinoise2(sim2, device_rates, observables)
        print("[PEC] SIMULATION_METHOD=paulinoise2 — paulinoise_simulator_2 + binomial shot noise")
    else:
        noise_model = build_aer_noise_model(device_rates, n_qubits)
        aer_sim = make_aer_sim(method="density_matrix", noise_model=noise_model, seed=seed)
        ideal_expectations, noisy_expectations = aer_reference_expectations(
            n_qubits, observables, cx_topology, noise_model, seed)
        print("[PEC] SIMULATION_METHOD=aer_dm — device noise from truth params")
    print(f"[PEC] Ideal ⟨O⟩ (noiseless): {ideal_expectations}")
    print(f"[PEC] Noisy ⟨O⟩ (unmitigated): {noisy_expectations}")

    all_results: dict = {}
    pec_costs: dict = {}
    for name in summaries:
        all_results[name] = {}
        for obs_str in observables:
            sampling = run_pec_sampling(
                params[name][:n_gate], obs_str, sample_sizes, seed,
                os.path.join(output_dir, f"pec_sampling_{name}_{obs_str}.json"), cx_topology,
            )
            total_cost = sampling["total_cost"]
            pec_costs.setdefault(name, total_cost)

            results = []
            for chk_idx, chk in enumerate(sampling["checkpoints"]):
                samples = sampling["samples_by_checkpoint"][chk]
                counter = Counter(s["gate_paulis"] for s in samples)
                batch_seed = seed + chk_idx * 10000
                meas = {}
                for j, key in enumerate(counter):
                    n_shots = max(1, counter[key] * per_instance_shots)
                    if method == "paulinoise2":
                        meas[key] = paulinoise2_measure_parity(
                            key, device_rates, obs_str, n_shots, sim2, np.random.default_rng(batch_seed + j))
                    else:
                        qc = build_circuit(key, n_qubits, obs_str, cx_topology)
                        job = aer_sim.run(qc, shots=n_shots, seed_simulator=batch_seed + j)
                        meas[key] = parity_expectation(job.result().get_counts(), obs_str)
                results.append(total_cost * sum(s["sign"] * meas[s["gate_paulis"]] for s in samples) / len(samples))
                total_shots = sum(max(1, c * per_instance_shots) for c in counter.values())
                print(f"[PEC] Checkpoint {chk}: {len(counter)} unique configs, "
                      f"count×{per_instance_shots} shots each → {total_shots} total shots")
            all_results[name][obs_str] = results
            print(f"[PEC] {name} / {obs_str}: {results}")

    out_json = os.path.join(output_dir, "pec_results.json")
    with open(out_json, "w") as f:
        json.dump({
            "checkpoints": sample_sizes,
            "observables": observables,
            "results": all_results,
            "ideal_expectations": ideal_expectations,
            "noisy_expectations": noisy_expectations,
            "pec_costs": pec_costs,
            "n_qubits": n_qubits,
            "cx_topology": cx_topology,
        }, f, indent=2)
    print(f"[PEC] Results saved → {out_json}")
    return all_results
