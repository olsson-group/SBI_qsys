"""
sbi_pec_zne.gpu_utils — Aer GPU detection and simulator factory.

Usage:
    from sbi_pec_zne.gpu_utils import make_aer_sim, GPU_AVAILABLE
"""
from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def _probe(method: str) -> bool:
    """
    Return True only if AerSimulator can *actually run* a circuit on GPU
    for the given method.  Simply checking available_devices() is not enough —
    some installations list GPU but fail at execution time.
    """
    try:
        sim = AerSimulator(method=method, device="GPU", precision="double")
        if "GPU" not in sim.available_devices():
            return False
        # Run a minimal 1-qubit circuit to confirm execution works.
        qc = QuantumCircuit(1)
        qc.h(0)
        if method == "density_matrix":
            qc.save_density_matrix()
        else:
            qc.save_statevector()
        sim.run(qc, shots=1).result()
        return True
    except Exception:
        return False


_GPU_DM    = _probe("density_matrix")
_GPU_SV    = _probe("statevector")
GPU_AVAILABLE: bool = _GPU_DM or _GPU_SV

if GPU_AVAILABLE:
    print(f"[GPU] Aer GPU available  density_matrix={_GPU_DM}  statevector={_GPU_SV}")


def make_aer_sim(
    method: str | None = None,
    noise_model=None,
    seed: int | None = None,
    **kwargs,
) -> AerSimulator:
    """
    Create an AerSimulator, enabling GPU only for methods that actually work.

    Parameters
    ----------
    method : 'density_matrix' | 'statevector' | None (auto)
    noise_model : Aer NoiseModel or None
    seed : seed_simulator value
    **kwargs : passed directly to AerSimulator
    """
    kw: dict = {}
    if method is not None:
        kw["method"] = method
    if noise_model is not None:
        kw["noise_model"] = noise_model
    if seed is not None:
        kw["seed_simulator"] = seed

    if GPU_AVAILABLE:
        is_dm = (method == "density_matrix")
        # Only enable GPU if we confirmed it works for this specific method.
        use_gpu = (is_dm and _GPU_DM) or (not is_dm and _GPU_SV)
        if use_gpu:
            kw.setdefault("device", "GPU")
            # GPU defaults to single precision (complex64); force double to match CPU (complex128)
            kw.setdefault("precision", "double")

    kw.update(kwargs)
    return AerSimulator(**kw)
