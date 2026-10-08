"""Pieces shared by the PEC and ZNE experiments: SBI parameters, circuits, Aer helpers."""

import os
from typing import Dict, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.noise import NoiseModel, pauli_error

from sbi_pec_zne.gpu_utils import make_aer_sim

# Index 0 is the identity; the other 15 follow the SBI parameter order of a CNOT.
PAULI_2Q = [
    "II", "IX", "IY", "IZ", "XI", "XX", "XY", "XZ",
    "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ",
]


def load_sbi_params(sbi_results_dir: str, ground_truth_dir: str) -> Dict[str, np.ndarray]:
    """Posterior mean and median of the SBI samples, and the true parameters.

    Reads ``paulinoise_inference_params.csv`` from ``sbi_results_dir`` and
    ``paulinoise_1_parameters.csv`` from ``ground_truth_dir``. The true parameters
    are the device noise of the experiments, so both files are required.
    """
    infer_csv = os.path.join(sbi_results_dir, "paulinoise_inference_params.csv")
    truth_csv = os.path.join(ground_truth_dir, "paulinoise_1_parameters.csv")
    try:
        samples = np.loadtxt(infer_csv, delimiter=",")
        truth = np.loadtxt(truth_csv, delimiter=",").flatten()
    except Exception as e:
        raise RuntimeError(f"Could not load SBI parameters ({infer_csv}, {truth_csv}): {e}") from e
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    print(f"Ground truth loaded ({len(truth)} params).")
    return {"mean": np.mean(samples, axis=0), "median": np.median(samples, axis=0), "truth": truth}


def cnot_order(n_cnots: int, cx_topology: str) -> list:
    """Bond indices in execution order (matches the paulinoise simulators)."""
    if cx_topology == "linear":
        return list(range(n_cnots))
    if cx_topology == "brickwall":
        return list(range(0, n_cnots, 2)) + list(range(1, n_cnots, 2))
    raise ValueError(f"cx_topology must be 'linear' or 'brickwall', got {cx_topology!r}")


def build_circuit(
    gate_paulis: tuple,
    n_qubits: int,
    observable_str: str,
    cx_topology: str = "brickwall",
    measure_for_shots: bool = True,
) -> QuantumCircuit:
    """Characterisation circuit with a fixed Pauli inserted after every CNOT.

    Prep Ry(π/4)·Rz(π/4) on all qubits, then the CNOTs in execution order. ``gate_paulis[g]`` is the
    2-character Pauli applied on (g, g+1) after CNOT g; all "II" gives the bare noisy circuit.
    With ``measure_for_shots`` the basis rotation for ``observable_str`` and ``measure_all()`` are
    appended; otherwise the circuit ends after the CNOTs (for density-matrix expectations).
    """
    n_cnots = n_qubits - 1
    if len(gate_paulis) != n_cnots:
        raise ValueError(f"Expected {n_cnots} gate Paulis, got {len(gate_paulis)}")
    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.ry(np.pi / 4, q)
        qc.rz(np.pi / 4, q)
    for g in cnot_order(n_cnots, cx_topology):
        ctrl, targ = g, g + 1
        qc.cx(ctrl, targ)
        p_str = gate_paulis[g]
        for qubit, p in ((ctrl, p_str[0]), (targ, p_str[1])):
            if p == "X":
                qc.x(qubit)
            elif p == "Y":
                qc.y(qubit)
            elif p == "Z":
                qc.z(qubit)
    if measure_for_shots:
        for q, p in enumerate(observable_str):
            if p == "X":
                qc.h(q)
            elif p == "Y":
                qc.sdg(q)
                qc.h(q)
        qc.measure_all()
    return qc


def parity_expectation(counts: dict, obs: str) -> float:
    """⟨O⟩ of a Pauli string from Z-basis counts (after the basis rotation): mean of (-1)^parity."""
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    exp = 0.0
    for bits, cnt in counts.items():
        parity = sum(int(b) for b, p in zip(bits[::-1], obs) if p != "I") % 2
        exp += cnt / total * (1 - 2 * parity)
    return exp


def build_aer_noise_model(truth_params: np.ndarray, n_qubits: int) -> NoiseModel:
    """Aer noise model with one Pauli channel per CNOT from the first (n-1)×15 gate parameters."""
    n_cnots = n_qubits - 1
    rates_matrix = truth_params[: n_cnots * 15].reshape(n_cnots, 15)
    noise_model = NoiseModel()
    for i in range(n_cnots):
        probs = rates_matrix[i].tolist()
        # Qiskit labels are little-endian, so each 2-qubit label is reversed
        error_list = [(label[::-1], p) for label, p in zip(PAULI_2Q[1:], probs)]
        error_list.append(("II", 1.0 - sum(probs)))
        noise_model.add_quantum_error(pauli_error(error_list), ["cx"], [i, i + 1])
    return noise_model


def dm_expectation(qc: QuantumCircuit, noise_model: Optional[NoiseModel], obs_str: str, seed: int) -> float:
    """Exact ⟨O⟩ of a (measurement-free) circuit from the Aer density matrix."""
    qc = qc.copy()
    qc.save_density_matrix()
    job = make_aer_sim(method="density_matrix", noise_model=noise_model, seed=seed).run(qc)
    dm = job.result().data()["density_matrix"]
    return float(dm.expectation_value(SparsePauliOp(obs_str[::-1])).real)


def aer_reference_expectations(
    n_qubits: int, observables: list, cx_topology: str, noise_model: NoiseModel, seed: int,
) -> tuple:
    """Ideal and noisy (no insertions, exact density matrix) ⟨O⟩ for each observable."""
    qc = build_circuit(("II",) * (n_qubits - 1), n_qubits, observables[0], cx_topology, measure_for_shots=False)
    ideal = {obs: dm_expectation(qc, None, obs, seed) for obs in observables}
    noisy = {obs: dm_expectation(qc, noise_model, obs, seed) for obs in observables}
    return ideal, noisy
