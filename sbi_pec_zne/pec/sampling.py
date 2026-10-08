"""Quasi-probability sampling for PEC: inverse Pauli channels and Pauli insertions."""

import json
import os
from datetime import datetime

import numpy as np
from qiskit.quantum_info import Pauli

from sbi_pec_zne.common import PAULI_2Q


def compute_inverse_pauli_channel(probs_vector):
    """Quasi-probabilities of the inverse of a 2-qubit Pauli channel.

    Args:
        probs_vector: the 15 non-identity error probabilities.

    Returns:
        (weights of the 15 non-identity Paulis, weight of the identity); they may be negative.
    """
    p_identity = 1.0 - np.sum(probs_vector)
    probs = np.concatenate(([p_identity], probs_vector))

    n_paulis = 16
    eigenvalues = np.zeros(n_paulis)
    for k in range(n_paulis):
        pk = Pauli(PAULI_2Q[k])
        eigenvalues[k] = sum(
            probs[j] * (1 if pk.commutes(Pauli(PAULI_2Q[j])) else -1)
            for j in range(n_paulis)
        )

    inv_eigs = 1.0 / eigenvalues
    inverse_weights = np.zeros(n_paulis)
    for k in range(n_paulis):
        pk = Pauli(PAULI_2Q[k])
        inverse_weights[k] = sum(
            inv_eigs[j] * (1 if pk.commutes(Pauli(PAULI_2Q[j])) else -1)
            for j in range(n_paulis)
        ) / n_paulis

    return inverse_weights[1:], inverse_weights[0]


def sample_pauli_combinations(rates_matrix: np.ndarray, max_samples: int, random_seed: int = 123):
    """Draw ``max_samples`` Pauli insertions (one 2-qubit Pauli per CNOT) from the inverse channels.

    Each CNOT's quasi-probabilities w are sampled with probability |w| / Σ|w|; a sample's sign is the
    product of the signs of the drawn weights. Seeds the global NumPy RNG.

    Returns (samples, total_cost) where each sample is {"gate_paulis", "sign", "sample_index"} and
    total_cost γ = Π_g Σ|w_g| is the sampling overhead.
    """
    weights_per_cnot = []
    gate_costs = []
    for rates in rates_matrix:
        q_quasi, q_id = compute_inverse_pauli_channel(rates)
        w = np.concatenate(([q_id], q_quasi))
        weights_per_cnot.append(w)
        gate_costs.append(np.sum(np.abs(w)))

    total_cost = float(np.prod(gate_costs))
    probs_per_cnot = [np.abs(w) / c for w, c in zip(weights_per_cnot, gate_costs)]

    np.random.seed(random_seed)
    samples = []
    for i in range(1, max_samples + 1):
        sign = 1.0
        gate_paulis = []
        for g, w in enumerate(weights_per_cnot):
            idx = np.random.choice(len(w), p=probs_per_cnot[g])
            sign *= np.sign(w[idx])
            gate_paulis.append(PAULI_2Q[idx])
        samples.append({"gate_paulis": tuple(gate_paulis), "sign": float(sign), "sample_index": i})
    return samples, total_cost


def run_pec_sampling(
    rates: np.ndarray,
    observable_str: str,
    sample_sizes: list,
    random_seed: int,
    output_file: str,
    cx_topology: str = "brickwall",
) -> dict:
    """Draw the quasi-probability samples once and truncate them to each checkpoint.

    ``rates`` are the 15 gate parameters of every CNOT, flattened ((n-1)×15). The samples are drawn
    up to the largest entry of ``sample_sizes`` (seed ``random_seed``) and checkpoint N uses the first
    N of them. The configuration is saved to ``output_file`` (json) and returned.
    """
    rates_matrix = np.asarray(rates).reshape(-1, 15)
    n_cnots = len(rates_matrix)
    checkpoints = sorted(int(s) for s in sample_sizes)
    max_samples = checkpoints[-1]

    print(f"[PEC] Quasi-prob sampling: drawing {max_samples} samples once (seed={random_seed}), "
          f"truncating to checkpoints {checkpoints}")
    samples, total_cost = sample_pauli_combinations(rates_matrix, max_samples, random_seed)
    samples_by_checkpoint = {chk: samples[:chk] for chk in checkpoints}

    data = {
        "observable": observable_str,
        "n_qubits": n_cnots + 1,
        "cx_topology": cx_topology,
        "n_cnots": n_cnots,
        "sample_sizes": sample_sizes,
        "checkpoints": checkpoints,
        "max_samples": max_samples,
        "random_seed": random_seed,
        "total_cost": total_cost,
        "samples_info": {"samples_by_checkpoint": {
            str(chk): [{"sample_index": s["sample_index"], "gate_paulis": list(s["gate_paulis"]), "sign": s["sign"]}
                       for s in samples_chk]
            for chk, samples_chk in samples_by_checkpoint.items()
        }},
        "timestamp": datetime.now().isoformat(),
    }
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[PEC] Sampling config saved → {output_file}")
    print(f"      max N={max_samples}, γ={total_cost:.4f}")
    return {"total_cost": total_cost, "checkpoints": checkpoints, "samples_by_checkpoint": samples_by_checkpoint}
