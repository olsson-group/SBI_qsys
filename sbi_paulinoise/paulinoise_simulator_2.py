"""
paulinoise_simulator_2.py — fast simulator for the sparse Pauli noise model (Stim + Pauli propagation).

The forward model SBI trains on: ``sbi_paulinoise`` simulates (θ, x) pairs with it, and its script
block writes such a dataset (``paulinoise_observables.csv``, ``paulinoise_2_parameters.csv``,
``paulinoise_2_observations.csv``). The ground truth comes from ``paulinoise_simulator_1.py``.

Model: same as simulator 1 (prep ``Ry(π/4), Rz(π/4)``, CNOTs ``brickwall`` or ``linear``, 15 Pauli
error rates per bond). Expectations are computed analytically by back-propagating Paulis through
the CNOTs, with Stim giving the Clifford sign; no density matrix, so it scales to much larger ``n``.

API: ``NoisyCircuitSimulator(n_qubits, cx_topology)`` with ``compute_expectations(params, observables)``
and ``compute_bitstring_probs(params)`` (exponential in ``n``); ``get_sbi_observables``,
``is_physical_paulinoise``, ``generate_and_save_data``.

Script block: edit ``N_QUBITS``, ``WEIGHTS``, priors, ``LEARNING_MODE``, ``CX_TOPOLOGY`` in
``if __name__ == '__main__'``; keep them consistent with ``paulinoise_simulator_1.py`` and
``sbi_paulinoise/run_sbi.py``.
"""

import itertools
import numpy as np
import stim
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

# ── Named default (same physics as Ry(π/4)·Rz(π/4)|0⟩ per qubit) ─────────────
STANDARD_PREP_STATES = {
    "ry_rz_pi4": "Ry(π/4)·Rz(π/4)|0⟩  — default characterisation state",
}

_BLOCH_BY_STATE = {
    "ry_rz_pi4": [0.5,  0.5,  1.0 / np.sqrt(2.0)],
}


def bloch_single_qubit_ry_rz(ry: float, rz: float) -> np.ndarray:
    """Bloch (⟨X⟩, ⟨Y⟩, ⟨Z⟩) for Ry(ry)·Rz(rz)|0⟩ (Qiskit order: Ry then Rz on |0⟩)."""
    return np.array(
        [
            np.sin(ry) * np.cos(rz),
            np.sin(ry) * np.sin(rz),
            np.cos(ry),
        ],
        dtype=np.float64,
    )


def prep_bloch_from_ry_rz_row(angles: np.ndarray, n_qubits: int) -> np.ndarray:
    """Build (n_qubits, 3) prep Bloch from flat [Ry0,Rz0,Ry1,Rz1,...]."""
    flat = np.asarray(angles, dtype=np.float64).ravel()
    if flat.size != 2 * n_qubits:
        raise ValueError(
            f"Expected {2 * n_qubits} Ry/Rz angles, got {flat.size}."
        )
    rows = [bloch_single_qubit_ry_rz(flat[2 * q], flat[2 * q + 1]) for q in range(n_qubits)]
    return np.stack(rows, axis=0)


def init_states_csv_header(n_qubits: int) -> List[str]:
    cols: List[str] = []
    for q in range(n_qubits):
        cols.extend([f"Ry{q}", f"Rz{q}"])
    return cols


def save_init_states_csv(path: str, init_ry_rz: np.ndarray, n_qubits: int) -> None:
    """Write paulinoise_init_states-style CSV: one row per initial product state."""
    arr = np.asarray(init_ry_rz, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != 2 * n_qubits:
        raise ValueError(f"init_ry_rz columns must be 2*n_qubits={2*n_qubits}, got {arr.shape[1]}")
    hdr = init_states_csv_header(n_qubits)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        for row in arr:
            w.writerow([f"{float(x):.17g}" for x in row])


def _parse_init_states_csv_rows(path: str) -> List[np.ndarray]:
    """Parse data rows into flat float arrays (no n_qubits check)."""
    with open(path) as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"Empty init states file: {path}")
    header0 = (rows[0][0] or "").strip().lower() if rows[0] else ""
    data_rows = rows[1:] if header0.startswith("ry") else rows
    flats: List[np.ndarray] = []
    for line in data_rows:
        if not line or all(not str(c).strip() for c in line):
            continue
        flats.append(np.array([float(c) for c in line], dtype=np.float64))
    if not flats:
        raise ValueError(f"No data rows in {path}")
    return flats


def load_init_states_ry_rz_rows(path: str, n_qubits: int) -> List[np.ndarray]:
    """Load CSV → list of length-(2*n_qubits) Ry/Rz angle rows (radians)."""
    flats = _parse_init_states_csv_rows(path)
    n_exp = 2 * n_qubits
    for i, flat in enumerate(flats):
        if flat.size != n_exp:
            raise ValueError(
                f"{path} row {i}: expected {n_exp} angles (Ry/Rz per qubit), got {flat.size}"
            )
    return flats


def load_init_states_prep_blochs(path: str, n_qubits: int) -> List[np.ndarray]:
    """Load CSV (header Ry0,Rz0,...) → list of (n_qubits, 3) Bloch tables."""
    flats = load_init_states_ry_rz_rows(path, n_qubits)
    return [prep_bloch_from_ry_rz_row(flat, n_qubits) for flat in flats]


class NoisyCircuitSimulator:
    """
    Simulates a quantum circuit with Ry(π/4), Rz(π/4) prep on each qubit + CNOT layer
    + Pauli noise, using the Heisenberg picture (Pauli propagation; Stim for CNOT signs).

    cx_topology options:
      "brickwall" – even sub-layer (0,1),(2,3),... then odd (1,2),(3,4),...
      "linear"    – sequential chain (0,1),(1,2),...,(n-2,n-1)
    """

    def __init__(self, n_qubits: int, cx_topology: str = "brickwall"):
        self.n_qubits = n_qubits
        self.n_cnots = n_qubits - 1
        self.cx_topology = cx_topology

        # Parameter counts: 15 two-qubit Pauli error rates per CNOT
        self.n_gate_params = self.n_cnots * 15
        self.total_params = self.n_gate_params

        # 15 non-identity 2-qubit Pauli labels (IX, IY, ..., ZZ)
        bases = ['I', 'X', 'Y', 'Z']
        self.pauli_labels_2q = [p1+p2 for p1 in bases for p2 in bases
                                 if not (p1 == 'I' and p2 == 'I')]
        assert len(self.pauli_labels_2q) == 15

        # CNOT pairs in forward execution order.
        # Both topologies share the same parameter index convention:
        #   gate_params[i] always corresponds to CX(i, i+1).
        #
        # True "brick-wall" schedule is implemented as two sub-layers:
        #   even sub-layer: (0,1),(2,3),(4,5),...
        #   odd  sub-layer: (1,2),(3,4),(5,6),...
        # and executed in that order, i.e. all even-CX first, then all odd-CX.
        if cx_topology == "linear":
            self._cnot_pairs = [(i, i+1) for i in range(self.n_cnots)]
        elif cx_topology == "brickwall":
            even_pairs = [(i, i+1) for i in range(0, self.n_cnots, 2)]
            odd_pairs = [(i, i+1) for i in range(1, self.n_cnots, 2)]
            self._cnot_pairs = even_pairs + odd_pairs
        else:
            raise ValueError(f"Unknown cx_topology '{cx_topology}'. Use 'brickwall' or 'linear'.")
        self._backprop_order  = list(reversed(self._cnot_pairs))

        # Precompute full back-propagation circuit (for sign tracking via stim).
        # One stim.Circuit object reused for all observables.
        self._full_backprop_circ = stim.Circuit()
        for ctl, tgt in self._backprop_order:
            self._full_backprop_circ.append("CNOT", [ctl, tgt])

        # Precompute symplectic (x, z) encodings of the 15 error Paulis for fast
        # vectorised commutation: I=(0,0), X=(1,0), Y=(1,1), Z=(0,1)
        _xz = {'I': (0, 0), 'X': (1, 0), 'Y': (1, 1), 'Z': (0, 1)}
        self._err_ex_c = np.array([_xz[lab[0]][0] for lab in self.pauli_labels_2q], dtype=np.uint8)
        self._err_ez_c = np.array([_xz[lab[0]][1] for lab in self.pauli_labels_2q], dtype=np.uint8)
        self._err_ex_t = np.array([_xz[lab[1]][0] for lab in self.pauli_labels_2q], dtype=np.uint8)
        self._err_ez_t = np.array([_xz[lab[1]][1] for lab in self.pauli_labels_2q], dtype=np.uint8)

    def compute_expectations(self, params_vector: np.ndarray, observables: List[str],
                             prep_bloch: np.ndarray = None) -> List[float]:
        """
        Compute analytical expectation values via Heisenberg back-propagation.

        Circuit: prep state on each qubit + CNOT layer (topology: self.cx_topology) + Pauli noise.

        Args:
            params_vector: Flattened noise parameter vector.
            observables:   List of n_qubits-length Pauli strings, e.g. ['IXYZ', ...].
            prep_bloch:    Per-qubit Bloch vectors, shape (n_qubits, 3), where each row
                           is [⟨X⟩, ⟨Y⟩, ⟨Z⟩] for qubit q in the (ideal) initial state.
                           None → default Ry(π/4)·Rz(π/4)|0⟩ state.
                           Use ``prep_bloch_from_ry_rz_row`` or ``load_init_states_prep_blochs``.

        Key optimisations vs naïve implementation:
          - Commutation signs computed with vectorised numpy (symplectic inner product)
            instead of creating stim.PauliString per error per CNOT per observable.
          - CNOT back-prop tracked with two uint8 arrays (symplectic x/z); stim called
            once per observable only for Clifford phase (sign) tracking.
        """
        if len(params_vector) != self.total_params:
            raise ValueError(f"Expected {self.total_params} params, got {len(params_vector)}.")

        # ---- Slice parameters ------------------------------------------------
        gate_params = params_vector[:self.n_gate_params].reshape(self.n_cnots, 15)

        # ---- Precompute per-CNOT identity probabilities ----------------------
        p_identity = 1.0 - gate_params.sum(axis=1)   # shape (n_cnots,)

        # ---- Prep-state Bloch vectors ----------------------------------------
        # _bloch[q] = [⟨X⟩_q, ⟨Y⟩_q, ⟨Z⟩_q] for the ideal (noise-free) initial state.
        if prep_bloch is None:
            _bloch = np.full((self.n_qubits, 3), [0.5, 0.5, np.cos(np.pi / 4)])
        else:
            _bloch = np.asarray(prep_bloch, dtype=np.float64)
            if _bloch.shape != (self.n_qubits, 3):
                raise ValueError(
                    f"prep_bloch must have shape ({self.n_qubits}, 3), got {_bloch.shape}."
                )

        results = []

        for obs_str in observables:
            if len(obs_str) != self.n_qubits:
                raise ValueError(f"Observable '{obs_str}' length ≠ {self.n_qubits} qubits.")

            # ---- 1. Init symplectic arrays from observable string ------------
            # Encoding: I=(0,0), X=(1,0), Y=(1,1), Z=(0,1)
            coeff = 1.0

            obs_x = np.zeros(self.n_qubits, dtype=np.uint8)
            obs_z = np.zeros(self.n_qubits, dtype=np.uint8)
            for q, pc in enumerate(obs_str):
                if pc == 'X':
                    obs_x[q] = 1
                elif pc == 'Y':
                    obs_x[q] = 1;  obs_z[q] = 1
                elif pc == 'Z':
                    obs_z[q] = 1

            # ---- 2. Back-propagation through CNOT pairs (reversed) ----------
            for ctl, tgt in self._backprop_order:
                probs = gate_params[ctl]   # indexed by ctrl qubit

                # A. Noise eigenvalue (vectorised symplectic commutation)
                #    anti[i] = 1 iff error i anti-commutes with current observable
                anti = ((obs_x[ctl] & self._err_ez_c) ^
                        (obs_z[ctl] & self._err_ex_c) ^
                        (obs_x[tgt] & self._err_ez_t) ^
                        (obs_z[tgt] & self._err_ex_t))          # shape (15,) uint8
                signs = 1 - 2 * anti.astype(np.float64)         # ±1
                coeff *= p_identity[ctl] + np.dot(probs, signs)

                # B. CNOT symplectic update (x_tgt ^= x_ctl, z_ctl ^= z_tgt)
                obs_x[tgt] ^= obs_x[ctl]
                obs_z[ctl] ^= obs_z[tgt]

            # ---- 3. Sign tracking via stim (one call per observable) --------
            sign_val = stim.PauliString(obs_str).after(self._full_backprop_circ).sign.real

            # ---- 4. Prep state evaluation (ideal) ---------------------------
            # Each factor is ⟨P_q⟩ in the initial state, using the Bloch vector.
            qubit_product = 1.0
            for q in range(self.n_qubits):
                xq, zq = int(obs_x[q]), int(obs_z[q])
                if xq == 0 and zq == 0:
                    continue                          # I → 1
                elif xq == 1 and zq == 0:
                    qubit_product *= _bloch[q, 0]     # ⟨X⟩_q
                elif xq == 1 and zq == 1:
                    qubit_product *= _bloch[q, 1]     # ⟨Y⟩_q
                else:
                    qubit_product *= _bloch[q, 2]     # ⟨Z⟩_q

            results.append(sign_val * qubit_product * coeff)

        return results

    def compute_bitstring_probs(self, params_vector: np.ndarray,
                                prep_bloch: np.ndarray = None) -> np.ndarray:
        """
        Compute Z-basis bitstring probability distribution p(b|θ) via
        Walsh-Hadamard transform on all 2^n Z-type Pauli expectations.

        p(b) = (1/2^n) Σ_{s=0}^{2^n-1} ⟨Z^s⟩(θ) · (-1)^{popcount(b & s)}

        Only Z-type Paulis (tensor products of I and Z) contribute to diagonal
        matrix elements ⟨b|ρ|b⟩, so X/Y expectations are irrelevant here.

        Args:
            prep_bloch: Per-qubit Bloch vectors (n_qubits, 3); see compute_expectations.
                        None → default Ry(π/4)·Rz(π/4) state.

        Returns: (2^n,) float32 array summing to 1.
        """
        n = self.n_qubits
        N = 2 ** n

        # Build all 2^n Z-type observable strings (s=0 → 'III...I' → identity)
        # Bit i of s set → qubit i has Z; ordering matches qubit 0 = leftmost char.
        z_obs = [
            ''.join('Z' if (s >> i) & 1 else 'I' for i in range(n))
            for s in range(1, N)   # skip s=0 (identity, always 1)
        ]

        exps = self.compute_expectations(params_vector, z_obs, prep_bloch=prep_bloch)

        # Coefficient vector c[s] = ⟨Z^s⟩; c[0] = 1 (identity)
        c = np.empty(N, dtype=np.float64)
        c[0] = 1.0
        c[1:] = exps

        # Fast Walsh-Hadamard transform: p = WHT(c) / N
        p = c.copy()
        h = 1
        while h < N:
            for i in range(0, N, h * 2):
                u = p[i:i + h].copy()
                v = p[i + h:i + 2 * h].copy()
                p[i:i + h]         = u + v
                p[i + h:i + 2 * h] = u - v
            h *= 2
        p /= N

        np.clip(p, 0.0, None, out=p)
        total = p.sum()
        if total > 0:
            p /= total
        return p.astype(np.float32)

def is_physical_paulinoise(params_vector: np.ndarray, n_qubits: int, max_noise_rate: float = 1.0) -> bool:
    """
    Check if the gate Pauli noise parameters are physically valid.

    Physical constraints:
    1. All probabilities must be non-negative
    2. For each CNOT gate, the sum of all error probabilities must be <= 1
    3. Each probability should be in [0, max_noise_rate]

    Args:
        params_vector: Flattened parameter vector of shape (n_cnots * 15,)
        n_qubits: Number of qubits
        max_noise_rate: Maximum allowed noise rate per parameter

    Returns:
        True if parameters are physically valid, False otherwise
    """
    n_cnots = n_qubits - 1
    if len(params_vector) != n_cnots * 15:
        return False

    gate_matrix = params_vector.reshape(n_cnots, 15)
    for i in range(n_cnots):
        probs = gate_matrix[i]
        if np.any(probs < 0) or np.any(probs > max_noise_rate):
            return False
        if np.sum(probs) > 1.0:
            return False
    return True

def get_sbi_observables(n_qubits: int = 8,
                        weights: List[int] = None,
                        n_random: int = None,
                        seed: int = None) -> List[str]:
    """
    Generate nearest-neighbor SBI characterization observables.

    Two modes:
      weights=[1,2,...] – return ALL NN observables for the given weights.
        Weight-w: all consecutive chains (i,...,i+w-1) with all 3^w non-identity combos.
        Count per weight w:  (n_qubits - w + 1) × 3^w
        Default weights=[1,2] → 87 strings for 8 qubits.

      weights=None – randomly sample n_random observables from the complete pool
        of ALL NN strings (weights 1 through n_qubits). Useful for exploring
        arbitrary observable subsets without fixing a weight structure.

    Big-endian convention: qubit 0 = leftmost character.

    Args:
        n_qubits:  number of qubits.
        weights:   list of Pauli weights, e.g. [1, 2] or [1, 3].
                   Pass None to enable random-sampling mode.
        n_random:  number of observables to sample when weights=None.
                   Required when weights=None.
        seed:      RNG seed for reproducibility (random-sampling mode only).
    """
    single = ['X', 'Y', 'Z']

    def _build_pool(weight_list: List[int]) -> List[str]:
        pool: List[str] = []
        for w in sorted(set(weight_list)):
            if not (1 <= w <= n_qubits):
                raise ValueError(f"Weight {w} out of range [1, {n_qubits}].")
            for i in range(n_qubits - w + 1):
                for combo in itertools.product(single, repeat=w):
                    s = ['I'] * n_qubits
                    for k, p in enumerate(combo):
                        s[i + k] = p
                    pool.append(''.join(s))
        return pool

    if weights is None:
        if n_random is None:
            raise ValueError("n_random must be specified when weights=None.")
        full_pool = _build_pool(list(range(1, n_qubits + 1)))
        rng = np.random.default_rng(seed)
        n_draw = min(n_random, len(full_pool))
        idx = sorted(rng.choice(len(full_pool), size=n_draw, replace=False))
        return [full_pool[i] for i in idx]

    return _build_pool(weights if weights != [] else [1, 2])


def _generate_single_sample_paulinoise(args: Tuple) -> Tuple[int, np.ndarray, np.ndarray, bool]:
    """Helper function for parallel processing of a single sample.

    Returns:
        (idx, x_sample, y_sample, is_valid)
        is_valid: True if sample satisfies physical constraints, False otherwise
    """
    idx, n_qubits, observables_list, noise_rate, seed, prior_type, prior_scale, reject_unphysical, max_noise_rate, learning_mode, cx_topology, init_prep_blochs = args
    rng = np.random.default_rng(seed + idx)
    sim = NoisyCircuitSimulator(n_qubits=n_qubits, cx_topology=cx_topology)

    # Generate parameters based on prior type
    if prior_type == "exponential":
        x_sample = rng.exponential(scale=prior_scale, size=sim.total_params)
    else:
        x_sample = rng.uniform(0, noise_rate, sim.total_params)

    # Check physical constraints (if enabled)
    is_valid = True
    if reject_unphysical:
        is_valid = is_physical_paulinoise(x_sample, n_qubits, max_noise_rate)
        if not is_valid:
            return idx, x_sample, None, False

    if init_prep_blochs is None:
        if learning_mode == "shots_learning":
            y_sample = sim.compute_bitstring_probs(x_sample).astype(np.float64)
        else:
            y_sample = np.array(sim.compute_expectations(x_sample, observables_list), dtype=np.float64)
    else:
        parts = []
        for bloch in init_prep_blochs:
            if learning_mode == "shots_learning":
                parts.append(sim.compute_bitstring_probs(x_sample, prep_bloch=bloch).astype(np.float64))
            else:
                parts.append(np.array(sim.compute_expectations(x_sample, observables_list, prep_bloch=bloch), dtype=np.float64))
        y_sample = np.concatenate(parts)
    return idx, x_sample, y_sample, is_valid

def generate_and_save_data(n_qubits, observables_list, n_samples, noise_rate=0.05, n_jobs=1,
                          prior_type="uniform", prior_scale=0.0005,
                          reject_unphysical=True, max_noise_rate=1.0,
                          learning_mode="obs_learning", cx_topology="brickwall",
                          init_prep_blochs=None):
    """
    Generates data and saves to CSV with optional parallel processing.

    Args:
        n_qubits: Number of qubits
        observables_list: List of observable strings
        n_samples: Number of samples to generate
        noise_rate: Upper bound for noise parameter sampling (for uniform prior)
        n_jobs: Number of parallel jobs (1 = serial, >1 = parallel, -1 = use all CPUs)
        prior_type: Type of prior distribution ("uniform" or "exponential")
        prior_scale: Scale parameter for Exponential prior (mean = scale)
        reject_unphysical: If True, reject samples that don't satisfy physical constraints
        max_noise_rate: Maximum allowed noise rate per parameter (for physical constraint check)
        learning_mode: "obs_learning" (expectations) or "shots_learning" (bitstring probs, 2^n_qubits-dim)
        init_prep_blochs: None → single default Ry(π/4)·Rz(π/4)|0⟩ per qubit.
            List of (n_qubits, 3) Bloch arrays → concatenate expectations for each prep;
            obs_dim = n_states × obs_dim_per_state. Use ``load_init_states_prep_blochs`` or
            ``prep_bloch_from_ry_rz_row`` to build from Ry/Rz angles.
    """
    np.random.seed(123)
    sim = NoisyCircuitSimulator(n_qubits=n_qubits, cx_topology=cx_topology)

    n_states = len(init_prep_blochs) if init_prep_blochs else 1
    n_out_per_state = 2 ** n_qubits if learning_mode == "shots_learning" else len(observables_list)
    n_out = n_out_per_state * n_states

    print("--- Simulation Config ---")
    print(f"Qubits: {n_qubits}")
    print(f"Samples: {n_samples}")
    print(f"Learning mode: {learning_mode}")
    if learning_mode == "obs_learning":
        print(f"Observables per state: {len(observables_list)}")
    else:
        print(f"Bitstring probs dim per state: {2 ** n_qubits}")
    if init_prep_blochs:
        print(f"Initial states ({n_states}): Ry/Rz product preps from init_prep_blochs")
        print(f"Total obs_dim: {n_out}  ({n_states} × {n_out_per_state})")
    else:
        print(f"Initial state: ry_rz_pi4 (default)  obs_dim: {n_out}")
    print(f"Prior type: {prior_type}")
    if prior_type == "exponential":
        print(f"  - Scale (mean): {prior_scale}")
    else:
        print(f"  - Range: [0, {noise_rate}]")

    # Pre-allocate arrays
    X_data = np.zeros((n_samples, sim.total_params))
    Y_data = np.zeros((n_samples, n_out))

    def _compute_y(x_sample):
        if init_prep_blochs is None:
            if learning_mode == "shots_learning":
                return np.array(sim.compute_bitstring_probs(x_sample), dtype=np.float64)
            return np.array(sim.compute_expectations(x_sample, observables_list), dtype=np.float64)
        parts = []
        for bloch in init_prep_blochs:
            if learning_mode == "shots_learning":
                parts.append(sim.compute_bitstring_probs(x_sample, prep_bloch=bloch).astype(np.float64))
            else:
                parts.append(np.array(sim.compute_expectations(x_sample, observables_list, prep_bloch=bloch), dtype=np.float64))
        return np.concatenate(parts)

    if n_jobs == 1:
        # Serial processing
        print("Generating data (serial)...")
        num_rejected = 0
        i = 0
        while i < n_samples:
            # 1. Generate random parameters based on prior type
            if prior_type == "exponential":
                x_sample = np.random.exponential(scale=prior_scale, size=sim.total_params)
            else:
                x_sample = np.random.uniform(0, noise_rate, sim.total_params)

            # 2. Check physical constraints (if enabled)
            if reject_unphysical:
                if not is_physical_paulinoise(x_sample, n_qubits, max_noise_rate):
                    num_rejected += 1
                    continue

            # 3. Compute observations
            y_sample = _compute_y(x_sample)
            if i == 0:
                print(f"Sample 0 Y values: {y_sample}")

            X_data[i, :] = x_sample
            Y_data[i, :] = y_sample
            i += 1

        if num_rejected > 0:
            print(f"Rejected {num_rejected} unphysical samples during generation.")
    else:
        # Parallel processing
        import multiprocessing as mp

        if n_jobs == -1:
            n_jobs = mp.cpu_count()

        seed = 123
        num_rejected = 0
        completed = 0
        max_attempts = n_samples * 10
        attempt = 0

        def _make_args(a):
            return (a, n_qubits, observables_list, noise_rate, seed + a, prior_type,
                    prior_scale, reject_unphysical, max_noise_rate, learning_mode,
                    cx_topology, init_prep_blochs)

        print(f"Generating data using {n_jobs} parallel workers...")
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {}
            next_slot = 0

            for _ in range(min(n_jobs * 2, max_attempts)):
                futures[executor.submit(_generate_single_sample_paulinoise, _make_args(attempt))] = attempt
                attempt += 1

            while completed < n_samples and attempt < max_attempts:
                if len(futures) == 0:
                    break

                for future in as_completed(futures):
                    attempt_idx = futures.pop(future)
                    try:
                        result_idx, x_sample, y_sample, is_valid = future.result()

                        if not is_valid:
                            num_rejected += 1
                            if completed < n_samples and attempt < max_attempts:
                                futures[executor.submit(_generate_single_sample_paulinoise, _make_args(attempt))] = attempt
                                attempt += 1
                        else:
                            if completed < n_samples:
                                X_data[next_slot, :] = x_sample
                                Y_data[next_slot, :] = y_sample
                                next_slot += 1
                                completed += 1

                                if completed == 1:
                                    print(f"Sample 0 Y values: {y_sample}")
                                if completed % 100 == 0:
                                    print(f"Progress: {completed}/{n_samples} (rejected: {num_rejected})")

                                if completed < n_samples and attempt < max_attempts:
                                    futures[executor.submit(_generate_single_sample_paulinoise, _make_args(attempt))] = attempt
                                    attempt += 1
                    except Exception as e:
                        print(f"Error processing sample {attempt_idx}: {e}")
                        num_rejected += 1
                        if completed < n_samples and attempt < max_attempts:
                            futures[executor.submit(_generate_single_sample_paulinoise, _make_args(attempt))] = attempt
                            attempt += 1

                    if completed >= n_samples:
                        break
        
        if completed < n_samples:
            print(f"Warning: Only generated {completed}/{n_samples} valid samples after {attempt} attempts.")
            print(f"Rejected {num_rejected} unphysical samples.")
            # Trim arrays to actual size
            X_data = X_data[:completed]
            Y_data = Y_data[:completed]
        else:
            print(f"Completed generating {n_samples} samples.")
            if num_rejected > 0:
                print(f"Rejected {num_rejected} unphysical samples during generation.")
 
    print("Saving files...")

    with open('paulinoise_observables.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Observable String"])
        for obs in observables_list:
            writer.writerow([obs])

    np.savetxt("paulinoise_2_parameters.csv", X_data, delimiter=",")
    np.savetxt("paulinoise_2_observations.csv", Y_data, delimiter=",")

    extra_init = ""
    if init_prep_blochs:
        extra_init = "\n - paulinoise_init_states.csv (Ry/Rz angles; keep with paulinoise_observables.csv for sim1/SBI)"

    print(
        f"Done. Files saved:\n - paulinoise_observables.csv\n - paulinoise_2_parameters.csv ({X_data.shape})\n"
        f" - paulinoise_2_observations.csv ({Y_data.shape}){extra_init}"
    )

# ==========================================
# ==========================================

if __name__ == "__main__":
    N_SAMPLES = 100  # number of (parameters, observations) rows to generate

    # Observable weights to include, e.g. [1, 2], [1, 3], [1, 2, 3], ...
    N_QUBITS = 5
    WEIGHTS = [1, 2,]
    n_random = None
    MY_OBSERVABLES = get_sbi_observables(N_QUBITS, weights=WEIGHTS, n_random=n_random)

    print(f"SBI observables: {len(MY_OBSERVABLES)} strings total")
    if WEIGHTS is None:
        print(f"  (random sampling: {len(MY_OBSERVABLES)} from full NN pool)")
    else:
        for w in sorted(set(WEIGHTS)):
            count = (N_QUBITS - w + 1) * 3**w
            print(f"  weight-{w}: {count} strings  ({N_QUBITS - w + 1} chains × 3^{w})")
    # Set n_jobs: 1 = serial, >1 = parallel, -1 = use all CPUs
    N_JOBS = -1  # Use all available CPUs

    # Prior configuration (should match sbi_paulinoise/run_sbi.py)
    PRIOR_TYPE = "uniform"  # Options: "uniform" or "exponential"
    PRIOR_SCALE = 0.001  # Scale parameter for Exponential prior (mean = scale)
    NOISE_RATE = 0.001  # Upper bound for uniform prior (if PRIOR_TYPE="uniform")

    # Rejection sampling configuration
    REJECT_UNPHYSICAL = True  # Reject samples that don't satisfy physical constraints
    MAX_NOISE_RATE = 1.0  # Maximum allowed noise rate per parameter (for physical constraint check)

    # Learning mode: "obs_learning" (Pauli expectations) or "shots_learning" (2^n-dim bitstring probs)
    LEARNING_MODE = "obs_learning"

    # CNOT topology of the characterization circuit — must match paulinoise_simulator_1.py
    CX_TOPOLOGY = "brickwall"    # Options: 'brickwall', 'linear'

    # Non-Clifford initial states: always write paulinoise_init_states.csv and use init_prep_blochs
    # (same path for N_INIT_STATES == 1 or > 1).
    INIT_STATES_CSV = "paulinoise_init_states.csv"
    # Single balanced non-Clifford state: ry=π/4, rz=π/4 → [0.500, 0.500, 0.707]
    # Multiple product states are mathematically redundant for gate-error characterisation
    # (coeff(θ) is state-independent; extra states only rescale observations).
    _angles = np.array([
        [np.pi / 4,   np.pi / 4],   # balanced non-Clifford
    ])  # shape (1, 2)
    _init_ry_rz = np.tile(_angles, (1, N_QUBITS))  # shape (1, 2*N_QUBITS)
    N_INIT_STATES = len(_init_ry_rz)
    save_init_states_csv(INIT_STATES_CSV, _init_ry_rz, N_QUBITS)
    _prep_list = [prep_bloch_from_ry_rz_row(_init_ry_rz[i], N_QUBITS) for i in range(N_INIT_STATES)]
    print(f"Wrote {INIT_STATES_CSV} with {N_INIT_STATES} Ry/Rz product state row(s).")

    generate_and_save_data(
        n_qubits=N_QUBITS,
        observables_list=MY_OBSERVABLES,
        n_samples=N_SAMPLES,
        noise_rate=NOISE_RATE,
        n_jobs=N_JOBS,
        prior_type=PRIOR_TYPE,
        prior_scale=PRIOR_SCALE,
        reject_unphysical=REJECT_UNPHYSICAL,
        max_noise_rate=MAX_NOISE_RATE,
        learning_mode=LEARNING_MODE,
        cx_topology=CX_TOPOLOGY,
        init_prep_blochs=_prep_list,
    )