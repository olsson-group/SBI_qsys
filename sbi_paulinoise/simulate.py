"""Parallel simulation workers and chunk-based runner.

Top-level functions are required for multiprocessing pickling.
"""

import sys
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sbi_paulinoise.paulinoise_simulator_2 import NoisyCircuitSimulator as _Sim2

# Bloch vectors: vertices of a regular tetrahedron inscribed in the unit sphere.
# All components = ±1/√3 ≈ ±0.577, so every (state, Pauli) pair gives a non-zero
# expectation value — unlike Clifford states which give 88% structural zeros.
_s = float(1.0 / np.sqrt(3))
_CANONICAL_BLOCH = np.array([
    [ _s,  _s,  _s],
    [ _s, -_s, -_s],
    [-_s,  _s, -_s],
    [-_s, -_s,  _s],
], dtype=np.float32)


def _sim_row(sim, row, observables, init_prep_blochs):
    """Simulate one theta row, returning a flat observation vector (all states × all Paulis)."""
    if init_prep_blochs is None:
        return np.array(sim.compute_expectations(row, observables), dtype=np.float32)
    parts = [
        np.array(sim.compute_expectations(row, observables, prep_bloch=bloch), dtype=np.float32)
        for bloch in init_prep_blochs
    ]
    return np.concatenate(parts)


def simulate_batch(args):
    """Process a contiguous chunk of theta rows. One _Sim2 per worker."""
    theta_chunk, n_qubits, observables, cx_topology, init_prep_blochs = args
    sim = _Sim2(n_qubits=n_qubits, cx_topology=cx_topology)
    return [_sim_row(sim, row, observables, init_prep_blochs) for row in theta_chunk]


def run_parallel(
    theta_np: np.ndarray,
    n_qubits: int,
    observables: list,
    cx_topology: str,
    n_jobs: int,
    desc: str = "Simulating",
    init_prep_blochs: list = None,
) -> np.ndarray:
    """Chunk theta_np into n_jobs slices and simulate in parallel.

    Returns x_np of shape (len(theta_np), n_obs).
    """
    if n_jobs == -1:
        import multiprocessing as _mp
        n_jobs = _mp.cpu_count()

    n_sims = len(theta_np)

    if n_jobs <= 1:
        sim = _Sim2(n_qubits=n_qubits, cx_topology=cx_topology)
        results = []
        with tqdm(total=n_sims, desc=f"  {desc}", leave=False) as pbar:
            for row in theta_np:
                results.append(_sim_row(sim, row, observables, init_prep_blochs))
                pbar.update(1)
        return np.array(results, dtype=np.float32)

    chunk_size = (n_sims + n_jobs - 1) // n_jobs
    chunk_indices = [
        (i * chunk_size, min((i + 1) * chunk_size, n_sims))
        for i in range(n_jobs)
        if i * chunk_size < n_sims
    ]
    chunks = [theta_np[s:e] for s, e in chunk_indices]
    task_args = [
        (chunk, n_qubits, observables, cx_topology, init_prep_blochs)
        for chunk in chunks
    ]
    chunk_results: list = [None] * len(chunks)

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        f2i = {executor.submit(simulate_batch, task_args[i]): i for i in range(len(chunks))}
        with tqdm(total=n_sims, desc=f"  {desc}", leave=False) as pbar:
            for fut in as_completed(f2i):
                i = f2i[fut]
                chunk_results[i] = fut.result()
                pbar.update(len(chunks[i]))

    return np.array(
        [row for chunk in chunk_results for row in chunk], dtype=np.float32
    )
