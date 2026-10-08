"""Analytic Clifford + Pauli-noise expectation values via Stim (Heisenberg picture)."""

from typing import List, Optional, Tuple

import numpy as np
import stim

from .constants import (
    _QISKIT_TO_STIM_MAP,
    _STIM_INVERSE_MAP,
    _QISKIT_NATIVE_CLIFFORD_TO_STIM,
    _ANTICOMM_MASK,
    _GATE_CIRC_CACHE,
    _CNOT_CIRC_CACHE,
)


def _qiskit_clifford_to_stim_gates(qc) -> List[Tuple[str, List[int], int]]:
    """Convert Clifford circuit to list of (stim_gate_name, qubits, quarter_turns)."""
    gates = []
    for inst in qc.data:
        name = inst.operation.name.lower()
        qubits = [qc.find_bit(q).index for q in inst.qubits]
        if name in ('rx', 'ry', 'rz'):
            angle_rad = float(inst.operation.params[0])
            qt = int(round(angle_rad / (np.pi / 2))) % 4
            if qt == 0:
                continue
            stim_name = _QISKIT_TO_STIM_MAP[(name, qt)]
            gates.append((stim_name, qubits, qt))
        elif name == 'cx':
            gates.append(('CNOT', qubits, 0))
        elif name in _QISKIT_NATIVE_CLIFFORD_TO_STIM:
            gates.append((_QISKIT_NATIVE_CLIFFORD_TO_STIM[name], qubits, 0))
    return gates


def _get_gate_circuit(gate_name: str, qubit: int) -> stim.Circuit:
    key = (gate_name, qubit)
    c = _GATE_CIRC_CACHE.get(key)
    if c is None:
        c = stim.Circuit(f"{gate_name} {qubit}")
        _GATE_CIRC_CACHE[key] = c
    return c


def _get_cnot_circuit(ctl: int, tgt: int) -> stim.Circuit:
    key = (ctl, tgt)
    c = _CNOT_CIRC_CACHE.get(key)
    if c is None:
        c = stim.Circuit(f"CNOT {ctl} {tgt}")
        _CNOT_CIRC_CACHE[key] = c
    return c


def _expectation_one_observable_stim(
    obs_str: str,
    gates: List[Tuple[str, List[int], int]],
    gate_params: np.ndarray,
    n_qubits: int,
    n_cnots: int,
    init_bloch: Optional[np.ndarray] = None,
) -> float:
    """One observable: back-propagate with Stim, apply Pauli noise at each CNOT.

    init_bloch: (n_qubits, 3) Bloch vectors [bx, by, bz] for a non-Clifford product
                initial state (Ry(theta)·Rz(phi)|0> per qubit). If None, uses |0...0>.
    """
    current_op = stim.PauliString(obs_str)
    coeff = 1.0

    for name, qubits, qt in reversed(gates):
        if name == 'CNOT':
            ctl, tgt = qubits[0], qubits[1]
            edge = min(ctl, tgt)
            if edge < n_cnots:
                pc = int(current_op[ctl])
                pt = int(current_op[tgt])
                coeff *= 1.0 - 2.0 * np.dot(gate_params[edge], _ANTICOMM_MASK[pc, pt])
            current_op = current_op.after(_get_cnot_circuit(ctl, tgt))
        else:
            inv_name = _STIM_INVERSE_MAP[name]
            current_op = current_op.after(_get_gate_circuit(inv_name, qubits[0]))

    sign_val = current_op.sign
    if hasattr(sign_val, 'real'):
        sign_val = sign_val.real
    sign_val = float(sign_val)

    if init_bloch is None:
        # Original: |0...0> initial state — nonzero only for pure Z-strings
        for q in range(n_qubits):
            p = int(current_op[q])
            if p == 1 or p == 2:
                return 0.0
        return sign_val * coeff
    else:
        # Non-Clifford product initial state: <P'> = prod_q <P'_q>_q
        val = 1.0
        for q in range(n_qubits):
            p = int(current_op[q])
            if p == 0:
                continue
            elif p == 1:        # X
                val *= init_bloch[q, 0]
            elif p == 2:        # Y
                val *= init_bloch[q, 1]
            else:               # Z
                val *= init_bloch[q, 2]
        return sign_val * coeff * val


def compute_expectations_clifford_stim(
    qc,
    params_vector: np.ndarray,
    observables: List[str],
    n_qubits: int,
    init_bloch: Optional[np.ndarray] = None,
) -> List[float]:
    """Simulator_2-style: analytical Clifford + Pauli noise via Stim.

    init_bloch: (n_qubits, 3) Bloch vectors for non-Clifford initial state.
                If None, uses |0...0> (original behavior).
    """
    n_cnots = n_qubits - 1
    n_gate = n_cnots * 15
    gate_params = params_vector[:n_gate].reshape(n_cnots, 15)
    gates = _qiskit_clifford_to_stim_gates(qc)
    return [
        float(_expectation_one_observable_stim(obs, gates, gate_params, n_qubits, n_cnots, init_bloch))
        for obs in observables
    ]
