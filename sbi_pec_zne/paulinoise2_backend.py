"""
sbi_pec_zne.paulinoise2_backend — paulinoise_simulator_2 backend for PEC / ZNE.

Replaces Qiskit Aer for circuit simulation.  Uses NoisyCircuitSimulator
(Heisenberg / Pauli back-propagation) to compute exact expectation values for
circuits that include deterministic Pauli insertions (as used in PEC quasi-prob
sampling and ZNE amplification).  Shot noise is then simulated via Multinomial
(Binomial) sampling.

Circuit model per CNOT g  (ctrl = ctl, tgt = ctl+1):
    CX_g  →  device_noise_Λ_g  →  Pauli_insertion_Q_g

Back-propagation (Heisenberg, reversed):
    1. Q_g     — sign from commutation with current observable
    2. Λ_g     — noise channel eigenvalue
    3. CX_g†   — CNOT symplectic update

Public API
----------
compute_expectations_with_insertions(sim, params, observables, gate_paulis) -> List[float]
paulinoise2_measure_parity(gate_paulis, truth_params, obs_str, n_shots, sim, rng) -> float
compute_ideal_paulinoise2(sim, observables) -> dict
compute_noisy_paulinoise2(sim, truth_params, observables) -> dict
make_paulinoise2_sim(n_qubits, cx_topology) -> NoisyCircuitSimulator
"""

from __future__ import annotations

import os
import sys
from typing import List

import numpy as np

# make the repository root importable
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sbi_paulinoise.paulinoise_simulator_2 import NoisyCircuitSimulator  # noqa: E402
import stim  # noqa: E402

# Symplectic (x, z) encoding for single-qubit Paulis
_XZ = {"I": (0, 0), "X": (1, 0), "Y": (1, 1), "Z": (0, 1)}


def make_paulinoise2_sim(n_qubits: int, cx_topology: str = "brickwall") -> NoisyCircuitSimulator:
    """Create a NoisyCircuitSimulator for PEC/ZNE use."""
    return NoisyCircuitSimulator(n_qubits, cx_topology=cx_topology)


def compute_expectations_with_insertions(
    sim: NoisyCircuitSimulator,
    params_vector: np.ndarray,
    observables: List[str],
    gate_paulis_insertions: tuple,
) -> List[float]:
    """
    Analytical expectation values for a circuit with deterministic Pauli insertions.

    Parameters
    ----------
    sim : NoisyCircuitSimulator
    params_vector : noise parameter vector (length sim.total_params)
    observables : list of n_qubits-length Pauli strings
    gate_paulis_insertions : tuple of length n_cnots; each entry is a 2-char
        Pauli string (e.g. "XI") for the insertion after CNOT(ctl, ctl+1).
        Indexed by ctrl-qubit index (same convention as gate_params / PEC sampling).
        Pass all-"II" for a bare noisy circuit with no insertions.
    """
    n = sim.n_qubits
    n_cnots = sim.n_cnots

    if len(params_vector) != sim.total_params:
        raise ValueError(f"Expected {sim.total_params} params, got {len(params_vector)}")
    if len(gate_paulis_insertions) != n_cnots:
        raise ValueError(
            f"Expected {n_cnots} Pauli insertions, got {len(gate_paulis_insertions)}"
        )

    gate_params = params_vector[: sim.n_gate_params].reshape(n_cnots, 15)
    p_identity = 1.0 - gate_params.sum(axis=1)

    # Precompute insertion (ex_c, ez_c, ex_t, ez_t) per ctrl-qubit index
    ins: dict = {}
    for ctl in range(n_cnots):
        q_str = gate_paulis_insertions[ctl]
        c_x, c_z = _XZ[q_str[0]]
        t_x, t_z = _XZ[q_str[1]]
        ins[ctl] = (c_x, c_z, t_x, t_z)

    # Default Bloch vectors: Ry(π/4)·Rz(π/4)|0⟩
    _bloch = np.full((n, 3), [0.5, 0.5, np.cos(np.pi / 4)])

    results: List[float] = []
    for obs_str in observables:
        coeff = 1.0

        obs_x = np.zeros(n, dtype=np.uint8)
        obs_z = np.zeros(n, dtype=np.uint8)
        for q, pc in enumerate(obs_str):
            if pc == "X":
                obs_x[q] = 1
            elif pc == "Y":
                obs_x[q] = 1
                obs_z[q] = 1
            elif pc == "Z":
                obs_z[q] = 1

        for ctl, tgt in sim._backprop_order:
            # Pauli insertion Q_g: sign is -1 iff O anti-commutes with Q_g on (ctl, tgt)
            c_x, c_z, t_x, t_z = ins[ctl]
            anti_ins = int(
                (obs_x[ctl] & c_z) ^ (obs_z[ctl] & c_x)
                ^ (obs_x[tgt] & t_z) ^ (obs_z[tgt] & t_x)
            )
            coeff *= 1 - 2 * anti_ins

            # noise channel eigenvalue
            probs = gate_params[ctl]
            anti = (
                (obs_x[ctl] & sim._err_ez_c)
                ^ (obs_z[ctl] & sim._err_ex_c)
                ^ (obs_x[tgt] & sim._err_ez_t)
                ^ (obs_z[tgt] & sim._err_ex_t)
            )
            signs = 1 - 2 * anti.astype(np.float64)
            coeff *= p_identity[ctl] + np.dot(probs, signs)

            # CNOT symplectic update
            obs_x[tgt] ^= obs_x[ctl]
            obs_z[ctl] ^= obs_z[tgt]

        # CNOT sign from stim
        sign_val = (
            stim.PauliString(obs_str).after(sim._full_backprop_circ).sign.real
        )

        # product of the single-qubit Bloch components of the prepared state
        qubit_product = 1.0
        for q in range(n):
            xq, zq = int(obs_x[q]), int(obs_z[q])
            if xq == 0 and zq == 0:
                continue
            elif xq == 1 and zq == 0:
                qubit_product *= _bloch[q, 0]
            elif xq == 1 and zq == 1:
                qubit_product *= _bloch[q, 1]
            else:
                qubit_product *= _bloch[q, 2]

        results.append(sign_val * qubit_product * coeff)

    return results


def paulinoise2_measure_parity(
    gate_paulis: tuple,
    truth_params: np.ndarray,
    obs_str: str,
    n_shots: int,
    sim: NoisyCircuitSimulator,
    rng: np.random.Generator,
) -> float:
    """
    Simulate a shot-based parity measurement using paulinoise_simulator_2 + Multinomial sampling.

    Computes the exact expectation value E = ⟨O⟩ for the circuit defined by
    `gate_paulis` (Pauli insertions) and `truth_params` (device noise), then
    draws `n_shots` Bernoulli samples from {+1, -1} to simulate measurement noise.

    Parameters
    ----------
    gate_paulis   : tuple of n_cnots 2-char Pauli strings (ctrl-qubit indexed)
    truth_params  : gate noise params, length (n_qubits-1)*15
    obs_str       : n_qubits-length Pauli string, e.g. "IZIIZIII"
    n_shots       : total shots to simulate
    sim           : NoisyCircuitSimulator
    rng           : numpy Generator for reproducibility

    Returns
    -------
    float: estimated ⟨O⟩ = (n_+ − n_−) / n_shots
    """
    if n_shots < 1:
        return 0.0
    E_exact = compute_expectations_with_insertions(
        sim, truth_params, [obs_str], gate_paulis
    )[0]
    p_plus = float(np.clip((1.0 + E_exact) / 2.0, 0.0, 1.0))
    n_plus = int(rng.binomial(n_shots, p_plus))
    return (n_plus - (n_shots - n_plus)) / n_shots


def compute_ideal_paulinoise2(
    sim: NoisyCircuitSimulator,
    observables: List[str],
) -> dict:
    """
    Ideal (noiseless) ⟨O⟩ via paulinoise_simulator_2: all-zero noise, no insertions.
    """
    all_ii = tuple("II" for _ in range(sim.n_cnots))
    zero_params = np.zeros(sim.total_params)
    vals = compute_expectations_with_insertions(sim, zero_params, observables, all_ii)
    return {obs: float(v) for obs, v in zip(observables, vals)}


def compute_noisy_paulinoise2(
    sim: NoisyCircuitSimulator,
    truth_params: np.ndarray,
    observables: List[str],
) -> dict:
    """
    Noisy (unmitigated, no insertions) ⟨O⟩ via paulinoise_simulator_2 with truth noise params.
    """
    all_ii = tuple("II" for _ in range(sim.n_cnots))
    vals = compute_expectations_with_insertions(sim, truth_params, observables, all_ii)
    return {obs: float(v) for obs, v in zip(observables, vals)}
