"""Training data: random Clifford Trotter circuits simulated analytically (Stim) with and without noise."""

import os
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import numpy as np

from sbi_dt.tools.simulation import compute_expectations_clifford_stim
from .circuit import build_clifford_trotter_circuit


def _clifford_worker(args: Tuple) -> Tuple[np.ndarray, np.ndarray, float]:
    """Ideal and noisy expectations of one random Clifford Trotter circuit (picklable for Pool)."""
    n_qubits, n_steps, circuit_seed, noise_params, observables, initial_non_clifford = args
    rng = np.random.default_rng(circuit_seed)
    init_angles = None
    if initial_non_clifford:
        init_angles = (rng.uniform(0.0, 2.0 * np.pi, n_qubits), rng.uniform(0.0, 2.0 * np.pi, n_qubits))
    qc, init_bloch = build_clifford_trotter_circuit(n_qubits, n_steps, rng, init_angles)
    n_gate = (n_qubits - 1) * 15
    ideal = compute_expectations_clifford_stim(qc, np.zeros(n_gate), observables, n_qubits, init_bloch=init_bloch)
    noisy = compute_expectations_clifford_stim(qc, noise_params, observables, n_qubits, init_bloch=init_bloch)
    return ideal, noisy, float(n_steps)


def generate_trotter_data(
    sbi_samples: np.ndarray,
    n_qubits: int,
    n_steps_list: List[int],
    observables: List[str],
    circuits_per_combo: int = 5,
    random_seed: int = 42,
    label: str = "train",
    n_workers: Optional[int] = None,
    initial_non_clifford: bool = False,
) -> Dict[str, np.ndarray]:
    """Noisy/ideal expectations of ``circuits_per_combo`` random Clifford circuits per Trotter depth.

    The noise is the mean of the SBI posterior ``sbi_samples``. With ``initial_non_clifford`` the
    circuits start from a random Ry·Rx product state (handled analytically) instead of |0⟩^N.
    Returns arrays ``noisy`` and ``ideal`` of shape (N, n_obs) and ``n_steps`` of shape (N,),
    with N = len(n_steps_list) × circuits_per_combo.
    """
    rng = np.random.default_rng(random_seed)
    n_obs = len(observables)
    N = len(n_steps_list) * circuits_per_combo
    n_workers = os.cpu_count() if (n_workers is None or n_workers <= 0) else n_workers

    noise_params = np.mean(sbi_samples, axis=0)
    if noise_params.size == 15:  # one 15-parameter channel shared by every CNOT
        noise_params = np.tile(noise_params, n_qubits - 1)
    noise_params = np.asarray(noise_params, dtype=np.float64)

    print(f"  [{label}] mean posterior noise, {len(n_steps_list)} depths × {circuits_per_combo} circuits (total={N})")
    if initial_non_clifford:
        print(f"  [{label}] initial_non_clifford=True: Ry·Rx product-state init (Stim Heisenberg + analytic Bloch)")

    tasks = [
        (n_qubits, n_steps, int(rng.integers(0, 2**31)), noise_params, observables, initial_non_clifford)
        for n_steps in n_steps_list
        for _ in range(circuits_per_combo)
    ]

    noisy_out = np.zeros((N, n_obs), dtype=np.float32)
    ideal_out = np.zeros((N, n_obs), dtype=np.float32)
    steps_out = np.zeros(N, dtype=np.float32)
    chunksize = max(1, N // (n_workers * 10))
    log_every = max(1, N // 10)
    with Pool(n_workers) as pool:
        for row, (ideal, noisy, ns) in enumerate(pool.imap(_clifford_worker, tasks, chunksize=chunksize)):
            ideal_out[row] = ideal
            noisy_out[row] = noisy
            steps_out[row] = ns
            if (row + 1) % log_every == 0:
                print(f"  [{label}] {row + 1}/{N} done")

    return {"noisy": noisy_out, "ideal": ideal_out, "n_steps": steps_out}
