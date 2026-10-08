"""Noisy Trotter circuits on a Qiskit Aer density-matrix backend (GPU when available)."""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Pauli


class TrotterNoisyRunner:
    """Runs Trotter circuits under the Pauli noise model of ``paulinoise_simulator_1``.

    Constructor kwargs:
      aer_device     Aer device ("CPU", "GPU", or "auto").
      aer_*          Aer parallelism options.
      gpu_batch_size Number of density matrices simulated per Aer job.
    """

    def __init__(
        self,
        n_qubits: int,
        noise_params: np.ndarray,
        random_seed: int = 123,
        cx_topology: str = "brickwall",
        aer_device: str = "CPU",
        aer_max_parallel_threads=None,
        aer_max_parallel_experiments=None,
        aer_executor_workers=None,
        aer_max_job_size=None,
        gpu_batch_size: int = 1,
    ):
        from qiskit_aer import AerSimulator
        from sbi_paulinoise.paulinoise_simulator_1 import NoisyCircuitSimulator
        from sbi_dt.tools.utils import _aer_gpu_available, _apply_aer_parallel_options

        self.n_qubits = n_qubits
        self.random_seed = random_seed
        self._aer_executor = None

        def _expand(p):
            p = np.asarray(p, dtype=np.float64).reshape(-1)
            exp = (n_qubits - 1) * 15
            if p.size == exp:    return p
            if p.size == 15:     return np.tile(p, n_qubits - 1)
            if p.size > exp:     return p[:exp]
            raise ValueError(f"noise_params size {p.size}, expected 15 or {exp}")

        sim = NoisyCircuitSimulator(n_qubits, random_seed=random_seed,
                                    cx_topology=cx_topology)
        sim_params = _expand(noise_params)
        self.noise_model = sim.create_noise_model(sim_params)
        self.gpu_batch_size = max(1, int(gpu_batch_size))

        if aer_device is None or aer_device.upper() == "AUTO":
            from sbi_dt.tools.utils import _detect_aer_gpu
            aer_device = "GPU" if _detect_aer_gpu() else "CPU"
            print(f"  TrotterRunner: auto-detected device={aer_device}")

        self.backend_dm = None
        if aer_device.upper() != "CPU":
            try:
                backend = AerSimulator(
                    method="density_matrix", device=aer_device, precision="double",
                    noise_model=self.noise_model, seed_simulator=random_seed,
                )
                if not _aer_gpu_available(backend):
                    raise RuntimeError("Aer GPU not supported on this system.")
                # max_parallel_threads / max_parallel_experiments are CPU knobs (0 = all cores
                # would force CPU execution), so only the executor options apply on GPU.
                _apply_aer_parallel_options(
                    backend,
                    aer_executor_workers=aer_executor_workers,
                    aer_max_job_size=aer_max_job_size,
                    runner_ref=self, label="TrotterRunner(Aer)",
                )
                self.backend_dm = backend
                print(f"  TrotterRunner(Aer): device={aer_device}")
            except Exception as e:
                print(f"  TrotterRunner(Aer): GPU failed ({e}), falling back to CPU")

        if self.backend_dm is None:
            self.backend_dm = AerSimulator(
                method="density_matrix", noise_model=self.noise_model, seed_simulator=random_seed,
            )
            if any(x is not None for x in (
                aer_max_parallel_threads, aer_max_parallel_experiments,
                aer_executor_workers, aer_max_job_size,
            )):
                _apply_aer_parallel_options(
                    self.backend_dm,
                    aer_max_parallel_threads=aer_max_parallel_threads,
                    aer_max_parallel_experiments=aer_max_parallel_experiments,
                    aer_executor_workers=aer_executor_workers,
                    aer_max_job_size=aer_max_job_size,
                    runner_ref=self, label="TrotterRunner(Aer)",
                )

    def _to_dm_circuits(self, circuits, scale_factor=1):
        """Convert Trotter circuits to Aer density-matrix circuits with optional ZNE folding."""
        n = self.n_qubits
        out = []
        for qc in circuits:
            qc_dm = QuantumCircuit(n)
            for inst in qc.data:
                name = inst.operation.name
                qargs = [qc.find_bit(q).index for q in inst.qubits]
                params = inst.operation.params
                if name == "rzz":
                    c, t = qargs
                    for _ in range(scale_factor):
                        qc_dm.cx(c, t)
                    if float(params[0]) != 0.0:
                        qc_dm.rz(float(params[0]), t)
                    for _ in range(scale_factor):
                        qc_dm.cx(c, t)
                elif name == "cx" and scale_factor > 1:
                    for _ in range(scale_factor):
                        qc_dm.cx(qargs[0], qargs[1])
                else:
                    qc_dm.append(inst.operation, qargs, inst.clbits)
            qc_dm.save_density_matrix()
            out.append(qc_dm)
        return out

    def _extract_expectations(self, dm, observables):
        """Compute per-observable expectations from a density matrix."""
        return [float(dm.expectation_value(Pauli(obs[::-1])).real) for obs in observables]

    def _run_chunked(self, qc_dm_list, observables, opt_level=1):
        """
        Transpile all circuits at once (amortises Python overhead), then
        submit in chunks of gpu_batch_size.  Each chunk's density matrices
        are extracted and freed before the next chunk starts.

        gpu_batch_size=1  → peak RAM = 1 DM (safe for any GPU/RAM).
        gpu_batch_size=2  → peak RAM = 2 DM (~34 GB for n=15), ~2× faster on A100 40 GB.
        """
        # Transpile the full list at once — much cheaper than T separate calls.
        transpiled_all = transpile(
            qc_dm_list, self.backend_dm,
            optimization_level=opt_level, seed_transpiler=self.random_seed,
        )
        results = []
        bs = self.gpu_batch_size
        for start in range(0, len(transpiled_all), bs):
            chunk = transpiled_all[start:start + bs]
            job = self.backend_dm.run(chunk).result()
            for i in range(len(chunk)):
                data_i = job.results[i].data
                dm = data_i["density_matrix"] if hasattr(data_i, "__getitem__") else getattr(data_i, "density_matrix")
                results.append(self._extract_expectations(dm, observables))
        return results

    def run_batch(self, circuits, observables):
        qc_dm_list = self._to_dm_circuits(circuits, scale_factor=1)
        return self._run_chunked(qc_dm_list, observables, opt_level=0)

    def run_batch_scaled(self, circuits, observables, scale_factor=1):
        assert scale_factor % 2 == 1, "scale_factor must be odd"
        qc_dm_list = self._to_dm_circuits(circuits, scale_factor=scale_factor)
        return self._run_chunked(qc_dm_list, observables, opt_level=0)
