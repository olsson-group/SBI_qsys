"""Trotter circuits for 1D TFIM dynamics: observables, exact circuit, random Clifford circuits."""

from typing import List, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector

_CLIFFORD_ANGLES = [0.0, np.pi / 2, np.pi, 3.0 * np.pi / 2]


def ry_rx_angles_to_bloch(ry_angles: np.ndarray, rx_angles: np.ndarray) -> np.ndarray:
    """Bloch vectors of Ry(θ)·Rx(φ)|0⟩ per qubit: (sin θ cos φ, -sin φ, cos θ cos φ). Shape (n_qubits, 3)."""
    bx = np.sin(ry_angles) * np.cos(rx_angles)
    by = -np.sin(rx_angles)
    bz = np.cos(ry_angles) * np.cos(rx_angles)
    return np.stack([bx, by, bz], axis=-1)


def get_tfim_observables(n_qubits: int) -> List[str]:
    """TFIM observables as big-endian Pauli strings (qubit 0 leftmost): X_i (n), then Z_i Z_{i+1} (n-1)."""
    n = n_qubits
    obs: List[str] = []
    for i in range(n):
        obs.append("I" * i + "X" + "I" * (n - i - 1))
    for i in range(n - 1):
        obs.append("I" * i + "ZZ" + "I" * (n - i - 2))
    return obs


def build_neel_trotter_circuit(
    n_qubits: int,
    J: float,
    h: float,
    dt: float,
    n_steps: int,
) -> QuantumCircuit:
    """First-order Trotter circuit for H = -J Σ Z_i Z_{i+1} - h Σ X_i from |0⟩^N.

    Each step: RZZ on even bonds, RZZ on odd bonds, then Rx on every qubit.
    """
    qc = QuantumCircuit(n_qubits)
    rzz_angle = -2.0 * J * dt
    rx_angle = -2.0 * h * dt
    for _ in range(n_steps):
        for i in range(0, n_qubits - 1, 2):
            qc.rzz(rzz_angle, i, i + 1)
        for i in range(1, n_qubits - 1, 2):
            qc.rzz(rzz_angle, i, i + 1)
        for i in range(n_qubits):
            qc.rx(rx_angle, i)
    return qc


def build_clifford_trotter_circuit(
    n_qubits: int,
    n_steps: int,
    rng: np.random.Generator,
    init_angles: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[QuantumCircuit, Optional[np.ndarray]]:
    """Random Clifford Trotter circuit (every RZZ and Rx angle drawn from {0, π/2, π, 3π/2}).

    Each step is [RZZ even bonds, RZZ odd bonds, Rx on every qubit], RZZ built from CX·Rz·CX.
    ``init_angles=(ry, rx)`` starts from the product state Ry·Rx|0⟩ (returned as ``init_bloch``,
    not part of the circuit); otherwise the start is |0⟩^N and ``init_bloch`` is None.
    """
    qc = QuantumCircuit(n_qubits)
    for _ in range(n_steps):
        for start in (0, 1):
            for i in range(start, n_qubits - 1, 2):
                theta = rng.choice(_CLIFFORD_ANGLES)
                qc.cx(i, i + 1)
                if theta != 0.0:
                    qc.rz(theta, i + 1)
                qc.cx(i, i + 1)
        for i in range(n_qubits):
            theta = rng.choice(_CLIFFORD_ANGLES)
            if theta != 0.0:
                qc.rx(theta, i)
    init_bloch = None
    if init_angles is not None:
        ry_angles, rx_angles = init_angles
        init_bloch = ry_rx_angles_to_bloch(np.asarray(ry_angles, dtype=np.float64),
                                           np.asarray(rx_angles, dtype=np.float64))
    return qc, init_bloch


def compute_ideal_expectations(qc: QuantumCircuit, observables: List[str]) -> List[float]:
    """Exact noiseless expectation values (statevector)."""
    sv = Statevector.from_instruction(qc)
    return [float(sv.expectation_value(Pauli(obs[::-1])).real) for obs in observables]
