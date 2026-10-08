"""Pure-state parameterisations (cholesky, brickwall circuit angles / MPS), fidelities and Pauli observables."""

from __future__ import annotations

import numpy as np

_PAULI_I = np.eye(2, dtype=np.complex128)
_PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_PAULI_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_LETTER = {"I": _PAULI_I, "X": _PAULI_X, "Y": _PAULI_Y, "Z": _PAULI_Z}


def pauli_string_matrix(s: str) -> np.ndarray:
    """Tensor product for a length-n Pauli string (left = qubit 0)."""
    m = np.array([[1.0 + 0.0j]])
    for ch in s:
        m = np.kron(m, _LETTER[ch])
    return m


def theta_to_psi(theta: np.ndarray) -> np.ndarray:
    """Interleaved Re/Im -> normalised |psi⟩ (d,) in complex128."""
    t = np.asarray(theta, dtype=np.float64).reshape(-1)
    d = t.size // 2
    z = t[0::2] + 1.0j * t[1::2]
    z = np.asarray(z, dtype=np.complex128).ravel()
    n2 = float(np.real(np.vdot(z, z)))
    nrm = np.sqrt(n2) if n2 > 0.0 and np.isfinite(n2) else 0.0
    if nrm < 1e-15 or not np.isfinite(nrm):
        out = np.zeros(d, dtype=np.complex128)
        out[0] = 1.0
        return out
    z = z / nrm
    if not np.all(np.isfinite(z.real)) or not np.all(np.isfinite(z.imag)):
        out = np.zeros(d, dtype=np.complex128)
        out[0] = 1.0
        return out
    return z


def psi_to_theta(psi: np.ndarray) -> np.ndarray:
    out = np.empty(2 * psi.size, dtype=np.float64)
    out[0::2] = np.real(psi)
    out[1::2] = np.imag(psi)
    return out


def rho_from_psi(psi: np.ndarray) -> np.ndarray:
    return np.outer(psi, np.conjugate(psi))


def mps_bond_schedule(n_qubits: int, chi: int) -> list[int]:
    r"""Open-boundary bond dimensions :math:`(D_0,\ldots,D_n)` with

    .. math::

        D_k = \min(2^k,\,2^{n-k},\,\chi),\quad D_0=D_n=1.
    """
    n = int(n_qubits)
    c = int(chi)
    if n <= 0:
        raise ValueError("n_qubits must be >= 1")
    if c <= 0:
        raise ValueError("chi must be >= 1")
    out: list[int] = []
    for k in range(n + 1):
        if k == 0 or k == n:
            out.append(1)
        else:
            out.append(min(2**k, 2 ** (n - k), c))
    return out


def mps_site_dims(n_qubits: int, chi: int) -> list[tuple[int, int]]:
    """Per-site ``(left, right)`` bond dims for site ``k``: ``(D_k, D_{k+1})``."""
    bonds = mps_bond_schedule(n_qubits, chi)
    n = int(n_qubits)
    return [(bonds[k], bonds[k + 1]) for k in range(n)]


def _pad_mps_site_tensor(ak: np.ndarray, left: int, right: int, phys: int = 2) -> np.ndarray:
    """Embed a site tensor into the fixed open-boundary bond layout ``(left, right, phys)``."""
    out = np.zeros((left, right, phys), dtype=np.complex128)
    l, r, p = ak.shape
    if l > left or r > right or p > phys:
        raise ValueError(f"Cannot pad {ak.shape} into ({left}, {right}, {phys})")
    out[:l, :r, :p] = ak
    return out


def brickwall_cnot_pair_lists(n_qubits: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Odd/even nearest-neighbour CNOT pairs (0-based), matching :func:`brickwall_cnot_unitary`."""
    n = int(n_qubits)
    odd_pairs: list[tuple[int, int]] = []
    i = 0
    while 2 * i + 1 < n:
        odd_pairs.append((2 * i, 2 * i + 1))
        i += 1
    even_pairs: list[tuple[int, int]] = []
    j = 0
    while 2 * j + 2 < n:
        even_pairs.append((2 * j + 1, 2 * j + 2))
        j += 1
    return odd_pairs, even_pairs


def linear_cnot_pairs(n_qubits: int) -> list[tuple[int, int]]:
    """Sequential nearest-neighbour CNOT chain: ``(0,1),(1,2),...,(n-2,n-1)``.

    Unlike :func:`brickwall_cnot_pair_lists` (odd sweep then even sweep, each pair
    disjoint within its sweep), this applies one CNOT at a time in order. Every bond
    is still touched exactly once by a gate that sees a currently-unentangled (product)
    cut, so — same as one brickwall layer — χ=2 is exact for a single layer at any
    ``n_qubits`` (see ``mps_apply_2q_gate_adjacent``).
    """
    n = int(n_qubits)
    return [(i, i + 1) for i in range(n - 1)]


def _cnot_pairs_for_topology(n_qubits: int, cnot_topology: str) -> list[tuple[int, int]]:
    t = str(cnot_topology).lower()
    if t == "linear":
        return linear_cnot_pairs(n_qubits)
    if t == "brickwall":
        odd_pairs, even_pairs = brickwall_cnot_pair_lists(n_qubits)
        return [*odd_pairs, *even_pairs]
    raise ValueError(f"Unknown cnot_topology={cnot_topology!r}; expected 'linear' or 'brickwall'.")


def _coerce_prep_angles(
    theta_y: float | np.ndarray,
    theta_x: float | np.ndarray,
    n_qubits: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Broadcast scalar or length-``n`` prep angles to per-qubit float64 vectors."""
    n = int(n_qubits)
    ty = np.broadcast_to(np.asarray(theta_y, dtype=np.float64), (n,)).copy()
    tx = np.broadcast_to(np.asarray(theta_x, dtype=np.float64), (n,)).copy()
    if ty.shape != (n,) or tx.shape != (n,):
        raise ValueError(
            f"theta_y/theta_x must be scalar or length-{n}; got shapes {ty.shape}, {tx.shape}"
        )
    return ty, tx


def mps_vacuum(n_qubits: int) -> list[np.ndarray]:
    """Open-boundary MPS for ``|0…0⟩``; each site tensor has shape ``(left, right, phys=2)``."""
    n = int(n_qubits)
    if n <= 0:
        raise ValueError("n_qubits must be >= 1")
    tensors: list[np.ndarray] = []
    for _ in range(n):
        ak = np.zeros((1, 1, 2), dtype=np.complex128)
        ak[0, 0, 0] = 1.0
        tensors.append(ak)
    return tensors


def mps_apply_1q_gate(tensors: list[np.ndarray], qubit: int, gate: np.ndarray) -> None:
    """Apply a single-qubit ``(2,2)`` gate on site ``qubit`` (in-place)."""
    q = int(qubit)
    u = np.asarray(gate, dtype=np.complex128)
    if u.shape != (2, 2):
        raise ValueError(f"expected 2×2 gate, got {u.shape}")
    tensors[q] = np.einsum("ps,lrs->lrp", u, tensors[q], optimize=True)


def mps_apply_2q_gate_adjacent(
    tensors: list[np.ndarray],
    site: int,
    gate: np.ndarray,
    *,
    chi_max: int,
) -> None:
    """Apply a ``(4,4)`` two-qubit gate on adjacent sites ``site`` and ``site+1`` (in-place)."""
    c = int(site)
    g = np.asarray(gate, dtype=np.complex128).reshape(4, 4)
    chi = max(1, int(chi_max))
    a = tensors[c]
    b = tensors[c + 1]
    l_dim, m_dim, _ = a.shape
    m2, r_dim, _ = b.shape
    if m_dim != m2:
        raise ValueError(f"MPS bond mismatch at site {c}: {m_dim} vs {m2}")

    theta = np.einsum("lms,mrt->lstr", a, b, optimize=True)
    theta_flat = theta.reshape(l_dim, 4, r_dim)
    theta = np.einsum("ab,lbr->lar", g, theta_flat, optimize=True)
    # Split composite physical index before SVD: (L,4,R) -> (L,2,2,R) -> (L*2,2*R).
    m_mat = theta.reshape(l_dim, 2, 2, r_dim).reshape(l_dim * 2, 2 * r_dim)
    u, svals, vh = np.linalg.svd(m_mat, full_matrices=False)
    chi_new = min(chi, u.shape[1], svals.shape[0], m_dim * 4, 4 * r_dim)
    chi_new = max(chi_new, 1)
    u = u[:, :chi_new]
    svals = svals[:chi_new]
    vh = vh[:chi_new, :]

    tensors[c] = u.reshape(l_dim, 2, chi_new).transpose(0, 2, 1)
    tensors[c + 1] = (np.diag(svals) @ vh).reshape(chi_new, 2, r_dim).transpose(0, 2, 1)


def mps_brickwall_circuit_tensors(
    theta: np.ndarray,
    *,
    n_qubits: int,
    bond_dim: int,
    ansatz_layers: int = 1,
    unified_ry_rx: bool = False,
    cnot_topology: str = "linear",
) -> list[np.ndarray]:
    """Multi-layer brickwall Rx/Ry + CNOT as open-boundary MPS site tensors."""
    n = int(n_qubits)
    chi = int(bond_dim)
    layers = split_brickwall_angle_layers(
        theta,
        n,
        ansatz_layers=ansatz_layers,
        unified_ry_rx=unified_ry_rx,
    )
    tensors = mps_vacuum(n)
    cnot_01 = _cnot_unitary(0, 1, 2)
    cnot_pairs = _cnot_pairs_for_topology(n, cnot_topology)
    for theta_y, theta_x in layers:
        ty, tx = _coerce_prep_angles(theta_y, theta_x, n)
        for q in range(n):
            mps_apply_1q_gate(tensors, q, _rx(tx[q]) @ _ry(ty[q]))
        for c, t in cnot_pairs:
            if t != c + 1:
                raise ValueError(f"brickwall MPS path requires adjacent pairs; got ({c}, {t})")
            mps_apply_2q_gate_adjacent(tensors, c, cnot_01, chi_max=chi)
    expected = mps_site_dims(n, chi)
    return [
        _pad_mps_site_tensor(ak, left, right, 2)
        for ak, (left, right) in zip(tensors, expected)
    ]


def num_params_brickwall_angles(
    n_qubits: int,
    *,
    per_qubit_prep: bool = False,
    unified_ry_rx: bool = False,
    ansatz_layers: int = 1,
) -> int:
    """Number of real latent parameters for the brickwall circuit ansatz.

    * shared (2L): one ``(θ_y, θ_x)`` pair per layer, broadcast to all qubits.
    * per-qubit split (2nL): independent ``θ_y[q], θ_x[q]`` per layer.
    * per-qubit unified (nL): ``Ry(θ_q) Rx(θ_q)`` per layer — Ry and Rx share ``θ_q``.
    """
    n = int(n_qubits)
    L = max(1, int(ansatz_layers))
    if bool(unified_ry_rx):
        if not bool(per_qubit_prep):
            raise ValueError("unified_ry_rx requires per_qubit_prep=True")
        return n * L
    if bool(per_qubit_prep):
        return 2 * n * L
    return 2 * L


def split_brickwall_angle_layers(
    theta: np.ndarray,
    n_qubits: int,
    *,
    ansatz_layers: int = 1,
    unified_ry_rx: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Decode θ into per-layer ``(theta_y, theta_x)`` vectors (length ``n`` each)."""
    t = np.asarray(theta, dtype=np.float64).reshape(-1)
    n = int(n_qubits)
    L = int(ansatz_layers)
    if L == 1:
        if t.size == 2:
            return [(np.full(n, t[0], dtype=np.float64), np.full(n, t[1], dtype=np.float64))]
        if bool(unified_ry_rx) and t.size == n:
            return [(t.copy(), t.copy())]
        if not unified_ry_rx and t.size == 2 * n:
            return [(t[:n].copy(), t[n:].copy())]
    elif L == 2:
        if t.size == 2:
            layer = (np.full(n, t[0], dtype=np.float64), np.full(n, t[1], dtype=np.float64))
            return [layer, layer]
        if t.size == 4:
            return [
                (np.full(n, t[0], dtype=np.float64), np.full(n, t[1], dtype=np.float64)),
                (np.full(n, t[2], dtype=np.float64), np.full(n, t[3], dtype=np.float64)),
            ]
        if bool(unified_ry_rx) and t.size == 2 * n:
            a, b = t[:n].copy(), t[n:].copy()
            return [(a, a), (b, b)]
        if not unified_ry_rx and t.size == 4 * n:
            return [
                (t[:n].copy(), t[n : 2 * n].copy()),
                (t[2 * n : 3 * n].copy(), t[3 * n :].copy()),
            ]
    raise ValueError(
        f"Unsupported brickwall theta width {t.size} for n={n}, ansatz_layers={L}, "
        f"unified_ry_rx={unified_ry_rx}"
    )


def sample_brickwall_angle_theta(
    n_qubits: int,
    size: int,
    rng: np.random.Generator,
    *,
    theta_y_range: tuple[float, float] = (-np.pi, np.pi),
    theta_x_range: tuple[float, float] = (-np.pi, np.pi),
    per_qubit_prep: bool = False,
    unified_ry_rx: bool = False,
    ansatz_layers: int = 1,
) -> np.ndarray:
    """Sample latent circuit angles for amortized circuit-ansatz tomography."""
    n = int(size)
    nq = int(n_qubits)
    L = max(1, int(ansatz_layers))
    y_lo, y_hi = float(theta_y_range[0]), float(theta_y_range[1])
    x_lo, x_hi = float(theta_x_range[0]), float(theta_x_range[1])
    width = num_params_brickwall_angles(
        nq,
        per_qubit_prep=per_qubit_prep,
        unified_ry_rx=unified_ry_rx,
        ansatz_layers=L,
    )
    if bool(unified_ry_rx):
        if not bool(per_qubit_prep):
            raise ValueError("unified_ry_rx requires per_qubit_prep=True")
        return rng.uniform(y_lo, y_hi, size=(n, width)).astype(np.float64, copy=False)
    if bool(per_qubit_prep):
        out = np.empty((n, width), dtype=np.float64)
        for layer in range(L):
            y = rng.uniform(y_lo, y_hi, size=(n, nq))
            x = rng.uniform(x_lo, x_hi, size=(n, nq))
            sl = slice(layer * 2 * nq, (layer + 1) * 2 * nq)
            out[:, sl] = np.concatenate([y, x], axis=1)
        return out
    out = np.empty((n, 2 * L), dtype=np.float64)
    for layer in range(L):
        out[:, 2 * layer] = rng.uniform(y_lo, y_hi, size=n)
        out[:, 2 * layer + 1] = rng.uniform(x_lo, x_hi, size=n)
    return out


def brickwall_theta_row_key(row: np.ndarray, *, decimals: int = 12) -> bytes:
    """Quantised θ row for exact dedup (``decimals`` rounded float64 bytes)."""
    q = np.round(np.asarray(row, dtype=np.float64).reshape(-1), int(decimals))
    return q.tobytes()


def brickwall_theta_keys(theta: np.ndarray, *, decimals: int = 12) -> set[bytes]:
    """Set of quantised keys for all rows in ``theta``."""
    arr = np.asarray(theta, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    dec = int(decimals)
    return {brickwall_theta_row_key(arr[i], decimals=dec) for i in range(arr.shape[0])}


def sample_brickwall_angle_theta_unique(
    n_qubits: int,
    size: int,
    rng: np.random.Generator,
    *,
    theta_y_range: tuple[float, float] = (-np.pi, np.pi),
    theta_x_range: tuple[float, float] = (-np.pi, np.pi),
    per_qubit_prep: bool = False,
    unified_ry_rx: bool = False,
    ansatz_layers: int = 1,
    forbidden_keys: set[bytes] | None = None,
    dedup_decimals: int = 12,
) -> tuple[np.ndarray, int]:
    """Sample brickwall angles with no duplicate rows (within draw or vs ``forbidden_keys``)."""
    n = int(size)
    nq = int(n_qubits)
    if unified_ry_rx and not per_qubit_prep:
        raise ValueError("unified_ry_rx requires per_qubit_prep=True")
    width = num_params_brickwall_angles(
        nq,
        per_qubit_prep=per_qubit_prep,
        unified_ry_rx=unified_ry_rx,
        ansatz_layers=ansatz_layers,
    )
    if n <= 0:
        return np.empty((0, width), dtype=np.float64), 0

    dec = int(dedup_decimals)
    seen: set[bytes] = set(forbidden_keys) if forbidden_keys else set()
    rows: list[np.ndarray] = []
    n_rejected = 0
    batch = max(64, min(n * 4, 8192))
    max_draws = max(n * 64, batch)
    n_drawn = 0

    while len(rows) < n:
        if n_drawn >= max_draws:
            raise RuntimeError(
                f"sample_brickwall_angle_theta_unique: only collected {len(rows)}/{n} unique rows "
                f"after {n_drawn} draws ({n_rejected} rejected)."
            )
        chunk = sample_brickwall_angle_theta(
            nq,
            batch,
            rng,
            theta_y_range=theta_y_range,
            theta_x_range=theta_x_range,
            per_qubit_prep=per_qubit_prep,
            unified_ry_rx=unified_ry_rx,
            ansatz_layers=ansatz_layers,
        )
        n_drawn += chunk.shape[0]
        for i in range(chunk.shape[0]):
            key = brickwall_theta_row_key(chunk[i], decimals=dec)
            if key in seen:
                n_rejected += 1
                continue
            seen.add(key)
            rows.append(chunk[i])
            if len(rows) >= n:
                break

    return np.stack(rows[:n], axis=0).astype(np.float64, copy=False), n_rejected


def theta_to_psi_brickwall_angles(
    theta: np.ndarray,
    n_qubits: int,
    *,
    ansatz_layers: int = 1,
    unified_ry_rx: bool = False,
    cnot_topology: str = "linear",
) -> np.ndarray:
    """Brickwall circuit-angle latent θ → dense normalised ``|ψ⟩``."""
    return brickwall_circuit_state_vector_from_theta(
        theta,
        n_qubits,
        ansatz_layers=ansatz_layers,
        unified_ry_rx=unified_ry_rx,
        cnot_topology=cnot_topology,
    )


def _apply_brickwall_layer_to_state_vector(
    psi: np.ndarray,
    n_qubits: int,
    theta_y: np.ndarray,
    theta_x: np.ndarray,
    *,
    cnot_topology: str = "linear",
) -> np.ndarray:
    """Apply one brickwall layer: ``R_x R_y`` on each qubit, then brickwall CNOTs."""
    n = int(n_qubits)
    ty, tx = _coerce_prep_angles(theta_y, theta_x, n)
    d = 2**n
    psi = np.asarray(psi, dtype=np.complex128).reshape(d)
    u_prep = np.array([[1.0 + 0.0j]], dtype=np.complex128)
    for q in range(n):
        u_prep = np.kron(u_prep, _rx(tx[q]) @ _ry(ty[q]))
    psi = u_prep @ psi
    return brickwall_cnot_unitary(n, cnot_topology=cnot_topology) @ psi


def brickwall_circuit_state_vector_from_theta(
    theta: np.ndarray,
    n_qubits: int,
    *,
    ansatz_layers: int = 1,
    unified_ry_rx: bool = False,
    cnot_topology: str = "linear",
) -> np.ndarray:
    """Dense ``|ψ⟩`` from multi-layer brickwall Rx/Ry + CNOT circuit angles."""
    n = int(n_qubits)
    psi = np.zeros(2**n, dtype=np.complex128)
    psi[0] = 1.0
    for theta_y, theta_x in split_brickwall_angle_layers(
        theta,
        n,
        ansatz_layers=ansatz_layers,
        unified_ry_rx=unified_ry_rx,
    ):
        psi = _apply_brickwall_layer_to_state_vector(psi, n, theta_y, theta_x, cnot_topology=cnot_topology)
    nrm = np.linalg.norm(psi)
    if nrm < 1e-15 or not np.isfinite(nrm):
        out = np.zeros(2**n, dtype=np.complex128)
        out[0] = 1.0
        return out
    return psi / nrm


def prep_to_theta(
    theta_y: float,
    theta_x: float,
    *,
    n_qubits: int,
    state_param: str,
    per_qubit_prep: bool = False,
    unified_ry_rx: bool = False,
    ansatz_layers: int = 1,
) -> np.ndarray:
    """θ of the brickwall state with prep angles ``(theta_y, theta_x)``, in the layout of ``state_param``.

    ``circuit_angles``: the angle vector (shared, per-qubit, or per-qubit unified; tiled over layers).
    ``cholesky``: the Re/Im encoding of the dense state (single layer, linear CNOT chain).
    """
    n = int(n_qubits)
    if state_param == "cholesky":
        return psi_to_theta(brickwall_rx_ry_state_vector(n, theta_y=theta_y, theta_x=theta_x))
    if state_param != "circuit_angles":
        raise ValueError(f"Unsupported state_param={state_param!r}; expected 'cholesky' or 'circuit_angles'")
    if unified_ry_rx and not per_qubit_prep:
        raise ValueError("unified_ry_rx requires per_qubit_prep=True")
    if unified_ry_rx:
        block = np.full(n, theta_y, dtype=np.float64)
    elif per_qubit_prep:
        block = np.concatenate([np.full(n, theta_y, dtype=np.float64), np.full(n, theta_x, dtype=np.float64)])
    else:
        block = np.asarray([theta_y, theta_x], dtype=np.float64)
    return np.tile(block, max(1, int(ansatz_layers)))


def theta_to_psi_and_rho(
    theta: np.ndarray,
    *,
    n_qubits: int,
    state_param: str,
    ansatz_layers: int = 1,
    unified_ry_rx: bool = False,
    cnot_topology: str = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    """Normalised |ψ⟩ and ρ = |ψ⟩⟨ψ| decoded from θ (``cholesky`` or ``circuit_angles``)."""
    if state_param == "cholesky":
        psi = theta_to_psi(theta)
    elif state_param == "circuit_angles":
        psi = theta_to_psi_brickwall_angles(
            theta,
            n_qubits=n_qubits,
            ansatz_layers=ansatz_layers,
            unified_ry_rx=unified_ry_rx,
            cnot_topology=cnot_topology,
        )
    else:
        raise ValueError(f"Unsupported state_param={state_param!r}; expected 'cholesky' or 'circuit_angles'")
    return psi, rho_from_psi(psi)


def theta_to_mps_tensors(
    theta: np.ndarray,
    n_qubits: int,
    bond_dim: int,
    *,
    ansatz_layers: int = 1,
    unified_ry_rx: bool = False,
    cnot_topology: str = "linear",
) -> list[np.ndarray]:
    """Brickwall-angle θ → MPS site tensors ``A_k`` with shape ``(left, right, phys=2)``."""
    return mps_brickwall_circuit_tensors(
        theta,
        n_qubits=n_qubits,
        bond_dim=bond_dim,
        ansatz_layers=ansatz_layers,
        unified_ry_rx=unified_ry_rx,
        cnot_topology=cnot_topology,
    )


def mps_norm_squared(tensors: list[np.ndarray]) -> float:
    r"""\ :math:`\langle\psi|\psi\rangle` for (possibly unnormalised) open-boundary MPS."""
    env = np.ones((1, 1), dtype=np.complex128)
    for ak in tensors:
        # env_new[r, u] = Σ_{l,m,s} conj(A[l,r,s]) env[l,m] A[m,u,s]
        env = np.einsum("l r s, l m, m u s -> r u", np.conjugate(ak), env, ak, optimize=True)
    return float(np.real(env[0, 0]))


def mps_overlap(tensors_bra: list[np.ndarray], tensors_ket: list[np.ndarray]) -> complex:
    r"""\ :math:`\langle\psi_B|\psi_A\rangle` via mixed transfer matrices (no dense :math:`2^n` state).

    Site layout matches :func:`theta_to_mps_tensors` — ``(left, right, phys)``.
    """
    if len(tensors_bra) != len(tensors_ket):
        raise ValueError("MPS chain lengths must match")
    env = np.ones((1, 1), dtype=np.complex128)
    for bk, ak in zip(tensors_bra, tensors_ket):
        # env_new[r_b, r_a] = Σ_{l_b,l_a,s} conj(B[l_b,r_b,s]) env[l_b,l_a] A[l_a,r_a,s]
        env = np.einsum("l r s, l m, m u s -> r u", np.conjugate(bk), env, ak, optimize=True)
    return complex(env[0, 0])


def fidelity_mps_tensors(
    tensors_a: list[np.ndarray],
    tensors_b: list[np.ndarray],
) -> float:
    r"""Pure-state fidelity :math:`|\langle\psi_B|\psi_A\rangle|^2 / (\|\psi_A\|^2\|\psi_B\|^2)`."""
    na = mps_norm_squared(tensors_a)
    nb = mps_norm_squared(tensors_b)
    if na < 1e-30 or nb < 1e-30 or not (np.isfinite(na) and np.isfinite(nb)):
        return 0.0
    ov = mps_overlap(tensors_b, tensors_a)
    return float(abs(ov) ** 2 / (na * nb))


def fidelity_pure(psi_a: np.ndarray, psi_b: np.ndarray) -> float:
    """Fidelity :math:`|\\langle\\psi_a|\\psi_b\\rangle|^2` (equals :math:`\\mathrm{Tr}(\\rho_a\\rho_b)` for pure states)."""
    a = np.asarray(psi_a, dtype=np.complex128).ravel()
    b = np.asarray(psi_b, dtype=np.complex128).ravel()
    return float(abs(np.vdot(a, b)) ** 2)


def _ry(theta: float) -> np.ndarray:
    """R_y(θ) = exp(-i θ Y / 2), column-vector convention."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _rx(theta: float) -> np.ndarray:
    """R_x(θ) = exp(-i θ X / 2), column-vector convention."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -1.0j * s], [-1.0j * s, c]], dtype=np.complex128)


def _cnot_unitary(control: int, target: int, n_qubits: int) -> np.ndarray:
    """CNOT with ``control``, ``target`` in 0..n-1 (same indexing as Pauli strings)."""
    d = 2**n_qubits
    op = np.zeros((d, d), dtype=np.complex128)
    for j in range(d):
        bits = [((j >> (n_qubits - 1 - q)) & 1) for q in range(n_qubits)]
        if bits[control] == 1:
            bits[target] ^= 1
        jp = 0
        for q in range(n_qubits):
            jp = (jp << 1) | bits[q]
        op[jp, j] = 1.0
    return op


def brickwall_cnot_unitary(n_qubits: int, *, cnot_topology: str = "linear") -> np.ndarray:
    r"""Brickwall/linear CNOT unitary (dense; only usable for small ``n_qubits``).

    ``cnot_topology="brickwall"`` (paper convention, 1-based labels in the docstring):

    .. math::

        U_{\mathrm{BW}} = \Big(\prod_k \mathrm{CNOT}_{2k,\,2k+1}\Big)
        \Big(\prod_k \mathrm{CNOT}_{2k-1,\,2k}\Big)

    with products only over pairs that exist for the given ``n_qubits``.  Qubit labels
    in the products are **1-based in the formula**; internally we use **0-based**
    indices (qubit ``1`` → ``0``, etc.).  As a matrix product ``U_{\mathrm{even}} @
    U_{\mathrm{odd}}``, the **odd** layer is applied to the ket **first**.

    For ``n_qubits = 6`` this is: odd pairs ``(1,2),(3,4),(5,6)`` then even
    ``(2,3),(4,5)`` in 1-based, i.e. ``(0,1),(2,3),(4,5)`` then ``(1,2),(3,4)`` in
    0-based.

    ``cnot_topology="linear"`` (default): a single sequential chain
    ``CNOT(0,1) -> CNOT(1,2) -> ... -> CNOT(n-2,n-1)``, applied to the ket in order.
    """
    d = 2**n_qubits
    cnot_pairs = _cnot_pairs_for_topology(n_qubits, cnot_topology)

    # Some BLAS backends on macOS can emit spurious floating warnings during matmul
    # even though these matrices are 0/1 permutation matrices. Silence them here
    # to keep visualization runs clean.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        U = np.eye(d, dtype=np.complex128)
        for c, t in cnot_pairs:
            U = _cnot_unitary(c, t, n_qubits) @ U
        return U


def product_rx_ry_vacuum_state(
    n_qubits: int,
    theta_y: float | np.ndarray = np.pi / 4.0,
    theta_x: float | np.ndarray = np.pi / 4.0,
) -> np.ndarray:
    r"""Product state :math:`\bigotimes_{q} R_x(\theta_x^{(q)}) R_y(\theta_y^{(q)}) |0\rangle`.

  Scalars broadcast to all qubits; pass length-``n`` arrays for per-qubit prep angles.
    Matrix product ``R_x @ R_y`` applies :math:`R_y` to the ket first, then :math:`R_x`.
    """
    n = int(n_qubits)
    ty, tx = _coerce_prep_angles(theta_y, theta_x, n)
    d = 2**n
    U_prep = np.array([[1.0 + 0.0j]], dtype=np.complex128)
    for q in range(n):
        U_prep = np.kron(U_prep, _rx(tx[q]) @ _ry(ty[q]))
    psi0 = np.zeros(d, dtype=np.complex128)
    psi0[0] = 1.0
    psi = U_prep @ psi0
    nrm = np.linalg.norm(psi)
    if nrm < 1e-15:
        return psi0
    return psi / nrm


def brickwall_rx_ry_state_vector(
    n_qubits: int,
    theta_y: float | np.ndarray = np.pi / 4.0,
    theta_x: float | np.ndarray = np.pi / 4.0,
) -> np.ndarray:
    r"""State :math:`U_{\mathrm{BW}} \bigotimes_q R_x(\theta_x^{(q)}) R_y(\theta_y^{(q)}) |0\rangle`.

    Scalars broadcast; length-``n`` arrays give independent prep angles per qubit.
    Uses the same :func:`brickwall_cnot_unitary` as the 15-qubit definition, truncated
    to ``n_qubits`` (e.g. ``n_qubits=6``).
    """
    psi = product_rx_ry_vacuum_state(n_qubits, theta_y, theta_x)
    U_bw = brickwall_cnot_unitary(n_qubits)
    psi = U_bw @ psi
    nrm = np.linalg.norm(psi)
    if nrm < 1e-15:
        return product_rx_ry_vacuum_state(n_qubits, theta_y, theta_x)
    return psi / nrm


def expectation_pauli_pure(psi: np.ndarray, pmat: np.ndarray) -> float:
    """⟨ψ|P|ψ⟩ for normalised |ψ⟩ — einsum avoids BLAS matmul (no spurious Accelerate warnings on macOS)."""
    psi = np.asarray(psi, dtype=np.complex128).ravel()
    if psi.size == 0:
        return 0.0
    if not np.all(np.isfinite(psi.real)) or not np.all(np.isfinite(psi.imag)):
        return 0.0
    P = np.asarray(pmat, dtype=np.complex128)
    if P.shape != (psi.size, psi.size):
        P = P.reshape(psi.size, psi.size)
    return float(np.real(np.einsum("i,ij,j->", psi.conj(), P, psi, optimize=False)))


def pauli_strings_weight1_nn(n_qubits: int) -> list[str]:
    """Same pattern as device_sbi: local X,Y,Z + nearest-neighbour two-qubit Paulis."""
    obs: list[str] = []
    for q in range(n_qubits):
        for p in "XYZ":
            obs.append("I" * q + p + "I" * (n_qubits - q - 1))
    for q in range(n_qubits - 1):
        for p1 in "XYZ":
            for p2 in "XYZ":
                obs.append("I" * q + p1 + p2 + "I" * (n_qubits - q - 2))
    return obs


def random_pauli_strings(
    n_qubits: int,
    n_operators: int,
    rng: np.random.Generator,
    *,
    exclude_all_identity: bool = True,
) -> list[str]:
    """Draw ``n_operators`` **pairwise distinct** random full Pauli strings (left = qubit 0).

    Each candidate is drawn by independent uniform choice of ``I,X,Y,Z`` on every
    qubit; if the string was already collected, it is discarded and redrawn.
    The list contains **no duplicates**. All-identity ``III...`` is skipped when
    ``exclude_all_identity`` is True.
    """
    max_non_identity = 4**n_qubits - (1 if exclude_all_identity else 0)
    if n_operators > max_non_identity:
        raise ValueError(
            f"n_operators={n_operators} exceeds distinct Paulis available ({max_non_identity}) "
            f"for n_qubits={n_qubits}"
        )
    letters = np.array(list("IXYZ"))
    seen: set[str] = set()
    out: list[str] = []
    max_tries = max(10_000, n_operators * 200)
    for _ in range(max_tries):
        if len(out) >= n_operators:
            break
        chars = rng.choice(letters, size=n_qubits, replace=True)
        s = "".join(chars.tolist())
        if exclude_all_identity and s == "I" * n_qubits:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    if len(out) < n_operators:
        raise RuntimeError(
            f"Could not draw {n_operators} distinct Pauli strings (got {len(out)})"
        )
    return out
