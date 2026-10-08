"""
paulinoise_simulator_1.py — ground truth for the sparse Pauli noise model (exact, Qiskit Aer).

Stands in for the real system: the script block draws true noise parameters and computes the
"observed" data, written to ``paulinoise_1_parameters.csv`` / ``paulinoise_1_observations.csv``
(``ground_truth_*q/``). The fast simulator SBI trains on is ``paulinoise_simulator_2.py``; both
implement the same model independently, so their agreement validates it.

Model: prep ``Ry(π/4), Rz(π/4)`` on every qubit, then CNOTs (``brickwall`` or ``linear``), each with
15 two-qubit Pauli error rates. Returns Pauli expectations or Z-basis bitstring probabilities.

API: ``NoisyCircuitSimulator(n_qubits, random_seed, cx_topology)`` with
``compute_expectations(params, observables)`` and ``compute_bitstring_probs(params)``
(density matrix; memory grows exponentially in qubits).

Script block: ``SIMULATION_METHOD`` is ``'aer_dm'`` (this file) or ``'sim2'`` (use when ``n`` is too
large for a density matrix; outputs keep the ``paulinoise_1_*`` names). Reads
``paulinoise_observables.csv`` produced by ``paulinoise_simulator_2.py``.
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Pauli
from qiskit_aer.noise import NoiseModel, pauli_error
import numpy as np
import csv
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for ``sbi_paulinoise`` imports

# ==========================================
# GPU Auto-detection
# ==========================================
try:
    _GPU_AVAILABLE = 'GPU' in AerSimulator().available_devices()
    if _GPU_AVAILABLE:
        print("[Aer] GPU device detected — density-matrix simulations will use GPU.")
except Exception:
    _GPU_AVAILABLE = False

def _make_aer_sim(**kwargs) -> AerSimulator:
    """Create AerSimulator, automatically selecting GPU when available."""
    if _GPU_AVAILABLE and 'device' not in kwargs:
        kwargs['device'] = 'GPU'
        # GPU defaults to single precision (complex64); force double to match CPU (complex128)
        kwargs.setdefault('precision', 'double')
    return AerSimulator(**kwargs)

def _default_prep_ry_rz_flat(n_qubits: int) -> np.ndarray:
    """Flat [Ry0,Rz0,...] = π/4 each — matches paulinoise_simulator_2 default prep."""
    a = np.empty(2 * n_qubits, dtype=np.float64)
    a[0::2] = np.pi / 4
    a[1::2] = np.pi / 4
    return a


def _apply_prep_ry_rz_product(qc: QuantumCircuit, flat: np.ndarray) -> None:
    """Apply Ry·Rz on each qubit from flat angles (radians), Qiskit order."""
    flat = np.asarray(flat, dtype=np.float64).ravel()
    nq = len(flat) // 2
    for q in range(nq):
        qc.ry(float(flat[2 * q]), q)
        qc.rz(float(flat[2 * q + 1]), q)


class NoisyCircuitSimulator:
    def __init__(self, n_qubits: int, random_seed: int = None, cx_topology: str = "brickwall"):
        """
        Args:
            cx_topology: CNOT connectivity for the characterization circuit.
                "brickwall" – even sub-layer (0,1),(2,3),... then odd (1,2),(3,4),...
                "linear"    – sequential chain (0,1),(1,2),...,(n-2,n-1)
        """
        self.n_qubits = n_qubits
        self.n_cnots = n_qubits - 1
        self.cx_topology = cx_topology

        # CNOT pairs in forward execution order. Both topologies share the parameter
        # convention gate_params[i] <-> CX(i, i+1); brickwall runs all even bonds first,
        # then all odd bonds.
        if cx_topology == "linear":
            self._cnot_pairs = [(i, i+1) for i in range(self.n_cnots)]
        elif cx_topology == "brickwall":
            even_pairs = [(i, i+1) for i in range(0, self.n_cnots, 2)]
            odd_pairs = [(i, i+1) for i in range(1, self.n_cnots, 2)]
            self._cnot_pairs = even_pairs + odd_pairs
        else:
            raise ValueError(f"Unknown cx_topology '{cx_topology}'. Use 'brickwall' or 'linear'.")

        self.n_gate_params = self.n_cnots * 15  # 15 two-qubit Pauli error rates per CNOT
        self.total_params = self.n_gate_params
        self.random_seed = random_seed

        # Order: IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ
        self.pauli_labels_2q = []
        bases = ['I', 'X', 'Y', 'Z']
        for p1 in bases:
            for p2 in bases:
                if p1 == 'I' and p2 == 'I': continue
                self.pauli_labels_2q.append(p1 + p2)
        self.label_to_idx = {label: i for i, label in enumerate(self.pauli_labels_2q)}

        seed_kw = {'seed_simulator': random_seed} if random_seed is not None else {}
        self.backend_dm = _make_aer_sim(method='density_matrix', **seed_kw)

    def create_noise_model(self, params_vector):
        """Construct the Qiskit NoiseModel (one Pauli channel per CNOT) from the gate parameters."""
        noise_model = NoiseModel()
        gate_params = params_vector[:self.n_gate_params].reshape(self.n_cnots, 15)

        # gate_params[ctrl] is the channel of CX(ctrl, ctrl+1), whatever the execution order.
        for (ctrl, tgt) in self._cnot_pairs:
            probs = gate_params[ctrl]
            error_list = []
            total_p = 0
            for p_idx, label in enumerate(self.pauli_labels_2q):
                p = probs[p_idx]
                if p > 0:
                    error_list.append((label[::-1], p))  # Qiskit is little-endian
                    total_p += p
            if total_p < 1.0:
                error_list.append(('II', 1.0 - total_p))
            noise_model.add_quantum_error(pauli_error(error_list), ['cx'], [ctrl, tgt])
        return noise_model

    def _prep_flat(self, prep_ry_rz_flat: Optional[np.ndarray]) -> np.ndarray:
        flat = (
            _default_prep_ry_rz_flat(self.n_qubits)
            if prep_ry_rz_flat is None
            else np.asarray(prep_ry_rz_flat, dtype=np.float64).ravel()
        )
        if flat.size != 2 * self.n_qubits:
            raise ValueError(f"prep_ry_rz_flat must have length {2 * self.n_qubits}, got {flat.size}.")
        return flat

    def _density_matrix(self, params_vector: np.ndarray, prep_ry_rz_flat: Optional[np.ndarray]):
        """Aer density matrix after prep + noisy CNOTs (no transpile: the circuit is ry, rz, cx only)."""
        if len(params_vector) != self.total_params:
            raise ValueError(
                f"Parameter size mismatch. Expected {self.total_params}, got {len(params_vector)}."
            )
        qc = QuantumCircuit(self.n_qubits)
        _apply_prep_ry_rz_product(qc, self._prep_flat(prep_ry_rz_flat))
        for ctrl, targ in self._cnot_pairs:
            qc.cx(ctrl, targ)
        qc.save_density_matrix()
        job = self.backend_dm.run(qc, noise_model=self.create_noise_model(params_vector))
        return job.result().data()['density_matrix']

    def compute_expectations(self, params_vector: np.ndarray, observables: List[str], method='aer_dm',
                             prep_ry_rz_flat: Optional[np.ndarray] = None) -> List[float]:
        """Exact Pauli expectations from the Aer density matrix.

        Args:
            observables: Pauli strings, qubit 0 first.
            prep_ry_rz_flat: Ry/Rz angles per qubit (length 2*n); None → π/4 each.
        """
        if method != 'aer_dm':
            raise ValueError(f"Unknown method: {method!r}. Only 'aer_dm' is supported.")
        dm = self._density_matrix(params_vector, prep_ry_rz_flat)
        return [dm.expectation_value(Pauli(obs_str[::-1])).real for obs_str in observables]

    def compute_bitstring_probs(self, params_vector: np.ndarray, method='aer_dm',
                                prep_ry_rz_flat: Optional[np.ndarray] = None) -> np.ndarray:
        """Z-basis bitstring probabilities p(b|θ) = ⟨b|ρ(θ)|b⟩, shape (2^n,), float32, sums to 1.

        Qiskit little-endian: bit i of the index is the outcome of qubit i. Same convention as
        paulinoise_simulator_2.compute_bitstring_probs.
        """
        if method != 'aer_dm':
            raise NotImplementedError(f"compute_bitstring_probs only supports 'aer_dm', got {method!r}.")
        dm = self._density_matrix(params_vector, prep_ry_rz_flat)
        probs = np.real(np.diag(dm.data)).copy()
        np.clip(probs, 0.0, None, out=probs)
        probs /= probs.sum()
        return probs.astype(np.float32)

def load_observables(filename='paulinoise_observables.csv'):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    obs_list = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader, None) 
        for row in reader:
            if row: obs_list.append(row[0])
    return obs_list


if __name__ == "__main__":
    # Observables come from paulinoise_simulator_2.py
    with open('paulinoise_observables.csv', 'r') as _f:
        observables = [row[0] for row in csv.reader(_f) if row and row[0] != "Observable String"]
    n_qubits = len(observables[0])
    CX_TOPOLOGY = 'brickwall'    # 'brickwall' or 'linear'
    RANDOM_SEED = 123
    np.random.seed(RANDOM_SEED)

    # 'sim2' → paulinoise_simulator_2 (Stim, no Aer); 'aer_dm' → this file (exact density matrix).
    SIMULATION_METHOD = 'sim2'

    # Ry/Rz initial product states, same CSV as paulinoise_simulator_2. None → single default prep.
    INIT_STATES_FILE = "paulinoise_init_states.csv"

    # Must match LEARNING_MODE in sbi_paulinoise/run_sbi.py and paulinoise_simulator_2.py.
    #   'obs_learning'   – Pauli expectations
    #   'shots_learning' – 2^n_qubits Z-basis bitstring probabilities
    LEARNING_MODE = 'obs_learning'

    if SIMULATION_METHOD == 'sim2':
        from sbi_paulinoise.paulinoise_simulator_2 import NoisyCircuitSimulator as PaulinoiseSim2Noisy
        sim = PaulinoiseSim2Noisy(n_qubits, cx_topology=CX_TOPOLOGY)
    elif SIMULATION_METHOD == 'aer_dm':
        sim = NoisyCircuitSimulator(n_qubits, random_seed=RANDOM_SEED, cx_topology=CX_TOPOLOGY)
    else:
        raise ValueError(f"Unknown SIMULATION_METHOD: {SIMULATION_METHOD}")

    print(f"Pauli observables: {len(observables)} strings for {n_qubits} qubits.")
    print(f"Number of CNOT gates: {sim.n_cnots}")
    print(f"Parameters: {sim.total_params}")

    # Noise rates uniform in [0, 0.001] for every gate parameter.
    np.random.seed(RANDOM_SEED)
    manual_gate_params = np.random.uniform(0.0, 0.001, size=(sim.n_cnots, 15))
    for i in range(sim.n_cnots):
        print(f"Gate {i} Total Error Rate: {np.sum(manual_gate_params[i]):.4f}")
    final_params_vector = manual_gate_params.flatten()

    def _run_one_state(prep_flat):
        if SIMULATION_METHOD == 'sim2':
            from sbi_paulinoise.paulinoise_simulator_2 import prep_bloch_from_ry_rz_row
            pb = None if prep_flat is None else prep_bloch_from_ry_rz_row(prep_flat, n_qubits)
            if LEARNING_MODE == 'shots_learning':
                return np.asarray(sim.compute_bitstring_probs(final_params_vector, prep_bloch=pb), dtype=np.float64)
            return sim.compute_expectations(final_params_vector, observables, prep_bloch=pb)
        if LEARNING_MODE == 'shots_learning':
            return sim.compute_bitstring_probs(final_params_vector, prep_ry_rz_flat=prep_flat)
        return sim.compute_expectations(final_params_vector, observables, prep_ry_rz_flat=prep_flat)

    n_out_per_state = 2**n_qubits if LEARNING_MODE == 'shots_learning' else len(observables)
    if INIT_STATES_FILE and os.path.isfile(INIT_STATES_FILE):
        from sbi_paulinoise.paulinoise_simulator_2 import load_init_states_ry_rz_rows
        states_to_run = load_init_states_ry_rz_rows(INIT_STATES_FILE, n_qubits)
        print(f"Loaded {len(states_to_run)} prep rows from {INIT_STATES_FILE}")
    else:
        states_to_run = [None]  # default π/4 prep
        if INIT_STATES_FILE:
            print(f"[Note] INIT_STATES_FILE={INIT_STATES_FILE!r} missing — using single default prep.")
    print(f"Learning mode: {LEARNING_MODE}  |  Method: {SIMULATION_METHOD}")
    print(f"Initial states: {len(states_to_run)} Ry/Rz product configuration(s)")
    print(f"obs_dim per state: {n_out_per_state}  →  total obs_dim: {n_out_per_state * len(states_to_run)}")

    parts = []
    for i, st in enumerate(states_to_run):
        print(f"  Running prep configuration {i + 1}/{len(states_to_run)}...")
        parts.append(np.array(_run_one_state(st), dtype=np.float64))
    y_values = np.concatenate(parts)
    print(f"Computed {len(y_values)} values total ({len(states_to_run)} states × {n_out_per_state})")

    X_to_save = final_params_vector.reshape(1, -1)
    Y_to_save = np.array(y_values).reshape(1, -1)
    np.savetxt("paulinoise_1_parameters.csv", X_to_save, delimiter=",")
    np.savetxt("paulinoise_1_observations.csv", Y_to_save, delimiter=",")
    print("\nDone.")
    print(f"Saved paulinoise_1_parameters.csv ({X_to_save.shape})")
    print(f"Saved paulinoise_1_observations.csv ({Y_to_save.shape})")
