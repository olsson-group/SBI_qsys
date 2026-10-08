"""Aer helpers shared by the Trotter runners: GPU detection and parallel options."""

import os
from concurrent.futures import ThreadPoolExecutor

from qiskit import QuantumCircuit, transpile


def _detect_aer_gpu() -> bool:
    """Return True if qiskit-aer-gpu is installed and a GPU device is available.

    Called at runner-creation time (not module import) so SLURM login-node vs
    compute-node differences don't cause a stale False result.
    """
    try:
        from qiskit_aer import AerSimulator
        avail = AerSimulator().available_devices()
        return 'GPU' in avail
    except Exception:
        return False

# Cached at import time for backwards-compat; runners also call _detect_aer_gpu()
# directly so a stale cached value doesn't lock them into CPU.
_AER_GPU_AVAILABLE: bool = _detect_aer_gpu()


def _aer_gpu_available(backend) -> bool:
    """Probe whether an AerSimulator backend actually supports GPU."""
    try:
        qc = QuantumCircuit(1)
        qc.save_density_matrix()
        job = backend.run(transpile(qc, backend))
        job.result()
        return True
    except RuntimeError as e:
        if "GPU" in str(e) or "not supported" in str(e).lower():
            return False
        raise
    except Exception:
        return False


def _apply_aer_parallel_options(
    backend, *,
    aer_max_parallel_threads=None,
    aer_max_parallel_experiments=None,
    aer_executor_workers=None,
    aer_max_job_size=None,
    runner_ref=None,
    label: str = "Aer",
):
    """Apply Aer parallel/threading options."""
    opts = {}
    if aer_max_parallel_threads is not None:
        opts["max_parallel_threads"] = int(aer_max_parallel_threads)
    if aer_max_parallel_experiments is not None:
        opts["max_parallel_experiments"] = int(aer_max_parallel_experiments)
    if opts:
        try:
            backend.set_options(**opts)
        except Exception as e:
            print(f"  {label}: set_options({opts}) failed: {e}")
    _nworkers = None
    if aer_executor_workers is not None:
        _n = int(aer_executor_workers)
        _nworkers = (os.cpu_count() or 4) if _n <= 0 else _n  # -1 or 0 = use all available
    if _nworkers is not None and _nworkers > 0:
        try:
            exc = ThreadPoolExecutor(max_workers=_nworkers)
            if runner_ref is not None and hasattr(runner_ref, "_aer_executor"):
                runner_ref._aer_executor = exc
            backend.set_options(executor=exc)
            backend.set_options(
                max_job_size=int(aer_max_job_size) if aer_max_job_size is not None else 1
            )
        except Exception as e:
            print(f"  {label}: executor (workers={_nworkers}) failed: {e}")
