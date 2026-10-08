"""Constants for the Stim simulation: Pauli labels, Qiskit/Stim gate maps, commutation tables."""

import numpy as np
from typing import Dict, Tuple

# 15 two-qubit Pauli labels, same order as paulinoise_simulator_1 / paulinoise_simulator_2
_PAULI_2Q_LABELS = [p1 + p2 for p1 in ['I', 'X', 'Y', 'Z'] for p2 in ['I', 'X', 'Y', 'Z'] if (p1, p2) != ('I', 'I')]

_QISKIT_TO_STIM_MAP = {
    ('rx', 1): 'SQRT_X', ('rx', 2): 'X', ('rx', 3): 'SQRT_X_DAG',
    ('ry', 1): 'SQRT_Y', ('ry', 2): 'Y', ('ry', 3): 'SQRT_Y_DAG',
    ('rz', 1): 'S', ('rz', 2): 'Z', ('rz', 3): 'S_DAG',
}

_STIM_INVERSE_MAP = {
    'SQRT_X': 'SQRT_X_DAG', 'SQRT_X_DAG': 'SQRT_X', 'X': 'X',
    'SQRT_Y': 'SQRT_Y_DAG', 'SQRT_Y_DAG': 'SQRT_Y', 'Y': 'Y',
    'S': 'S_DAG', 'S_DAG': 'S', 'Z': 'Z', 'CNOT': 'CNOT', 'H': 'H',
}

_QISKIT_NATIVE_CLIFFORD_TO_STIM: Dict[str, str] = {
    'h': 'H', 's': 'S', 'sdg': 'S_DAG', 'x': 'X', 'y': 'Y', 'z': 'Z',
}

# Single-qubit commutation: +1 if commute, -1 if anticommute. Index: 0=I, 1=X, 2=Y, 3=Z
_COMM_1Q = np.array([
    [+1, +1, +1, +1],
    [+1, +1, -1, -1],
    [+1, -1, +1, -1],
    [+1, -1, -1, +1],
], dtype=np.int8)

_P2I = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}

_ERROR_IDX_PAIRS = np.array([(_P2I[l[0]], _P2I[l[1]]) for l in _PAULI_2Q_LABELS], dtype=np.int8)

_ANTICOMM_MASK = np.zeros((4, 4, 15), dtype=np.float64)
for _pc in range(4):
    for _pt in range(4):
        for _i, (_ec, _et) in enumerate(_ERROR_IDX_PAIRS):
            _sign = _COMM_1Q[_pc, _ec] * _COMM_1Q[_pt, _et]
            _ANTICOMM_MASK[_pc, _pt, _i] = 1.0 if _sign == -1 else 0.0

_GATE_CIRC_CACHE: Dict[Tuple[str, int], object] = {}
_CNOT_CIRC_CACHE: Dict[Tuple[int, int], object] = {}
