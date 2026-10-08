"""Exact Aer statevector reference for the Rydberg Trotter circuit (used only by benchmark_pp_rydberg.py)."""

from __future__ import annotations

import numpy as np


def _build_interactions(
    pos: np.ndarray,
    n: int,
    c6: float,
    scale: float,
    v_thr: float,
) -> list[tuple[int, int, float]]:
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            R = max(float(np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])), 1e-6)
            Vij = scale * c6 / R**6
            if abs(Vij) >= v_thr:
                out.append((i, j, float(Vij)))
    return out



def _build_circuit(
    n: int,
    Omega: float,
    Delta: float,
    interactions: list[tuple[int, int, float]],
    dt: float,
    n_steps: int,
    snap_steps: list[int],
    z_perturb: int | None = None,
):
    """Trotter circuit with save_statevector snapshots.

    Initial state: |+...+⟩ = H^⊗n |0...0⟩.
    If z_perturb given, apply Z on that qubit after the H layer.

    Per-step unitary  U_step = Rx · Rz · RZZ = e^{-iH_X dt} e^{-iH_Z dt} e^{-iH_ZZ dt},
    i.e. the interaction RZZ layer is applied first to the state. Qiskit applies
    gates in insertion order, so we add them RZZ → Rz → Rx.
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n)
    for i in range(n):
        qc.h(i)
    if z_perturb is not None:
        qc.z(z_perturb)

    snap_set = set(snap_steps)
    rx = 2.0 * Omega * dt
    rz = 2.0 * Delta * dt

    for step in range(n_steps + 1):
        if step in snap_set:
            qc.save_statevector(label=f"sv_{step}")
        if step == n_steps:
            break
        for i, j, Vij in interactions:
            qc.rzz(2.0 * Vij * dt, i, j)
        if Delta != 0.0:
            for i in range(n):
                qc.rz(rz, i)
        for i in range(n):
            qc.rx(rx, i)

    return qc


def _expect_pauli(sv: np.ndarray, ops: list[tuple[int, str]], dim: int) -> float:
    """⟨ψ| ⊗_k P_k |ψ⟩  for ops = [(qubit, 'X'|'Y'|'Z'), ...]."""
    idx = np.arange(dim, dtype=np.int64)
    flip_mask = 0
    phase = np.ones(dim, dtype=np.complex128)
    for q, p in ops:
        z = 1.0 - 2.0 * ((idx >> q) & 1).astype(np.float64)
        if p == "X":
            flip_mask ^= (1 << q)
        elif p == "Y":
            flip_mask ^= (1 << q)
            phase *= -1j * z
        elif p == "Z":
            phase *= z
    return float(np.real(np.vdot(sv, phase * sv[idx ^ flip_mask])))
