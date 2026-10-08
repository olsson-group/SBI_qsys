"""Simulator: pure-state θ → Pauli expectation vector ⟨P⟩ ∈ [-1, 1] (optionally shot-noisy).

``state_param``:
  · ``cholesky``       θ = interleaved Re/Im of |ψ⟩ (dense; CUDA vectorised or NumPy)
  · ``circuit_angles`` θ = brickwall Rx/Ry angles; dense |ψ⟩ for small n, MPS transfer-matrix
                       contraction (``tn_avoid_dense=True``) for large n, on CUDA or CPU
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from tqdm import tqdm

from sbi_qst.physics import (
    _cnot_pairs_for_topology,
    expectation_pauli_pure,
    pauli_string_matrix,
    theta_to_mps_tensors,
    theta_to_psi_brickwall_angles,
)


def _sample_pauli_expectation_torch(
    exp: torch.Tensor,
    shots: int | None,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Return Pauli expectation ⟨P⟩ ∈ [-1, 1].

    shots None  → exact expectation.
    shots int   → shot-noisy estimate: outcomes are ±1 with P(+1)=0.5(1+⟨P⟩),
                  so the empirical mean is 2·Binomial(shots, p+)/shots − 1.
    """
    n_shots = None if shots is None else int(shots)
    if n_shots is None:
        return exp.to(dtype=torch.float32)
    p_plus = (0.5 * (1.0 + exp)).clamp_(0.0, 1.0)
    freq = torch.binomial(
        torch.full_like(p_plus, float(n_shots)),
        p_plus,
        generator=generator,
    ) / float(n_shots)
    return (2.0 * freq - 1.0).to(dtype=torch.float32)


def _sample_pauli_expectation_numpy(
    rng: np.random.Generator,
    exp: np.ndarray,
    shots: int | None,
) -> np.ndarray:
    """NumPy counterpart of :func:`_sample_pauli_expectation_torch`."""
    n_shots = None if shots is None else int(shots)
    if n_shots is None:
        return np.asarray(exp, dtype=np.float32)
    p_plus = np.clip(0.5 * (1.0 + np.asarray(exp)), 0.0, 1.0)
    freq = rng.binomial(n_shots, p_plus) / float(n_shots)
    return (2.0 * freq - 1.0).astype(np.float32)


def _build_pauli_stack(pauli_strings: list[str]) -> np.ndarray:
    """Return (n_pauli, d, d) complex64 array of Pauli matrices."""
    return np.stack([pauli_string_matrix(s) for s in pauli_strings], axis=0).astype(np.complex64)


def _theta_batch_to_psi(theta_np: np.ndarray) -> np.ndarray:
    """(N, 2d) float → (N, d) complex128, each row normalised.

    Rows with zero / non-finite norm are replaced by |0⟩.
    """
    t = theta_np.astype(np.float64)
    z = t[:, 0::2] + 1j * t[:, 1::2]          # (N, d) complex128
    norms = np.linalg.norm(z, axis=1, keepdims=True)  # (N, 1)
    bad_norm = (norms < 1e-15).ravel()
    norms = np.where(norms < 1e-15, 1.0, norms)
    psi = z / norms
    # Kill rows that are non-finite after division
    bad = bad_norm | ~np.all(np.isfinite(psi.real) & np.isfinite(psi.imag), axis=1)
    if bad.any():
        psi[bad] = 0.0
        psi[bad, 0] = 1.0
    return psi  # (N, d) complex128


def _simulate_gpu(
    theta: "np.ndarray | torch.Tensor",
    P_stack_np: np.ndarray,
    shots: int,
    seed: int,
    chunk_size: int = 50_000,
) -> np.ndarray:
    """GPU-accelerated vectorised simulation. Returns (N, n_pauli) float32.

    theta may be a CPU numpy array or a GPU torch tensor (avoids H2D round-trip
    when the caller already has theta on device, e.g. from prior.sample()).

    For each chunk B:
      ψ[b]     — normalised state vector (complex64), computed on-device from θ
      exp[b,p] = Re(⟨ψ[b]|P[p]|ψ[b]⟩)   via batched einsum
      obs[b,p] = Pauli expectation ⟨P⟩ ∈ [-1,1] (shot-noisy when shots is set)
    """
    device = torch.device("cuda")
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    P_stack = torch.as_tensor(P_stack_np, dtype=torch.complex64, device=device)  # (n_pauli, d, d)
    n_pauli = P_stack.shape[0]

    # Bring theta to GPU once — use pinned memory when source is a numpy array.
    if isinstance(theta, np.ndarray):
        theta_gpu = torch.as_tensor(theta, dtype=torch.float32).pin_memory().to(device, non_blocking=False)
    else:
        theta_gpu = theta.to(device=device, dtype=torch.float32)

    N = theta_gpu.shape[0]
    out_gpu = torch.empty((N, n_pauli), dtype=torch.float32, device=device)

    for start in tqdm(range(0, N, chunk_size), desc="GPU sim", leave=False):
        end = min(start + chunk_size, N)
        B = end - start
        chunk = theta_gpu[start:end]  # (B, 2d) float32, already on device

        # θ→ψ on-device: interleaved Re/Im → complex64, then normalise
        z = torch.view_as_complex(chunk.reshape(B, -1, 2).contiguous())  # (B, d)
        norms = torch.linalg.norm(z, dim=1, keepdim=True)
        bad = (norms < 1e-15).squeeze(1)
        psi = z / norms.clamp(min=1e-15)                                  # (B, d)
        if bad.any():
            psi = psi.clone()
            psi[bad] = 0
            psi[bad, 0] = 1

        # P[p,d,j] · ψ[b,j] → Ppsi[b,p,d]
        Ppsi = torch.einsum("pdj,bj->bpd", P_stack, psi)                 # (B, n_pauli, d)
        # ⟨ψ|P|ψ⟩ = Σ_d ψ*[b,d] · Ppsi[b,p,d]
        exp = torch.real(
            torch.einsum("bd,bpd->bp", psi.conj(), Ppsi)
        ).clamp(-1.0, 1.0)                                                # (B, n_pauli)
        out_gpu[start:end] = _sample_pauli_expectation_torch(exp, shots, generator=gen)

    return out_gpu.cpu().numpy().astype(np.float32)


def _cnot_permutation_torch(n_qubits: int, control: int, target: int, device: torch.device) -> torch.Tensor:
    n = int(n_qubits)
    c = int(control)
    t = int(target)
    idx = torch.arange(2**n, device=device, dtype=torch.long)
    control_bit = (idx >> (n - 1 - c)) & 1
    perm = idx ^ (control_bit << (n - 1 - t))
    return perm


def _brickwall_angle_layers_torch(
    theta: torch.Tensor,
    n_qubits: int,
    *,
    ansatz_layers: int,
    unified_ry_rx: bool,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Batched decode of brickwall θ into per-layer angle vectors."""
    th = theta.to(dtype=torch.float32)
    n = int(n_qubits)
    L = int(ansatz_layers)
    if L == 1:
        if th.shape[1] == 2:
            return [(th[:, 0:1].expand(-1, n), th[:, 1:2].expand(-1, n))]
        if unified_ry_rx and th.shape[1] == n:
            return [(th, th)]
        if not unified_ry_rx and th.shape[1] == 2 * n:
            return [(th[:, :n], th[:, n:])]
    elif L == 2:
        if th.shape[1] == 2:
            layer = (th[:, 0:1].expand(-1, n), th[:, 1:2].expand(-1, n))
            return [layer, layer]
        if th.shape[1] == 4:
            return [
                (th[:, 0:1].expand(-1, n), th[:, 1:2].expand(-1, n)),
                (th[:, 2:3].expand(-1, n), th[:, 3:4].expand(-1, n)),
            ]
        if unified_ry_rx and th.shape[1] == 2 * n:
            return [(th[:, :n], th[:, :n]), (th[:, n:], th[:, n:])]
        if not unified_ry_rx and th.shape[1] == 4 * n:
            return [
                (th[:, :n], th[:, n : 2 * n]),
                (th[:, 2 * n : 3 * n], th[:, 3 * n :]),
            ]
    raise ValueError(
        f"brickwall theta width {th.shape[1]} unsupported for n={n}, "
        f"ansatz_layers={L}, unified_ry_rx={unified_ry_rx}"
    )


def _rx_ry_gate_torch(theta_y: torch.Tensor, theta_x: torch.Tensor) -> torch.Tensor:
    """Per-qubit ``R_x(θ_x) R_y(θ_y)`` as complex ``(B, n, 2, 2)``."""
    cy = torch.cos(0.5 * theta_y)
    sy = torch.sin(0.5 * theta_y)
    cx = torch.cos(0.5 * theta_x)
    sx = torch.sin(0.5 * theta_x)
    j = torch.tensor(1.0j, device=theta_y.device, dtype=torch.complex64)
    ry = torch.stack(
        [torch.stack([cy, -sy], dim=-1), torch.stack([sy, cy], dim=-1)],
        dim=-2,
    ).to(torch.complex64)
    rx = torch.stack(
        [torch.stack([cx, -j * sx], dim=-1), torch.stack([-j * sx, cx], dim=-1)],
        dim=-2,
    ).to(torch.complex64)
    return torch.matmul(rx, ry)


def _apply_brickwall_cnots_mps_torch(
    tensors: list[torch.Tensor],
    *,
    n_qubits: int,
    bond_dim: int,
    cnot_topology: str = "linear",
) -> list[torch.Tensor]:
    n = int(n_qubits)
    chi = max(1, int(bond_dim))
    th = tensors[0]
    cnot = torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        device=th.device,
        dtype=torch.complex64,
    )
    cnot_pairs = _cnot_pairs_for_topology(n, cnot_topology)

    def _apply_adjacent_cnot(site: int) -> None:
        a = tensors[site]
        b = tensors[site + 1]
        _, l_dim, m_dim, _ = a.shape
        _, m2, r_dim, _ = b.shape
        if m_dim != m2:
            raise RuntimeError(f"MPS bond mismatch at site {site}: {m_dim} vs {m2}")
        pair = torch.einsum("b l m s, b m r t -> b l s t r", a, b)
        pair_flat = pair.reshape(a.shape[0], l_dim, 4, r_dim)
        pair = torch.einsum("a c, b l c r -> b l a r", cnot, pair_flat)
        mat = pair.reshape(a.shape[0], l_dim, 2, 2, r_dim).reshape(a.shape[0], l_dim * 2, 2 * r_dim)
        u, svals, vh = torch.linalg.svd(mat, full_matrices=False)
        chi_new = max(1, min(chi, u.shape[2], svals.shape[1], m_dim * 4, 4 * r_dim))
        u = u[:, :, :chi_new]
        svals = svals[:, :chi_new]
        vh = vh[:, :chi_new, :]
        tensors[site] = u.reshape(a.shape[0], l_dim, 2, chi_new).permute(0, 1, 3, 2).contiguous()
        tensors[site + 1] = (svals[:, :, None] * vh).reshape(a.shape[0], chi_new, 2, r_dim).permute(0, 1, 3, 2).contiguous()

    for c, t in cnot_pairs:
        if t != c + 1:
            raise ValueError(f"brickwall MPS path requires adjacent pairs; got ({c}, {t})")
        _apply_adjacent_cnot(c)
    return tensors


def _theta_batch_to_psi_brickwall_angles_torch(
    theta: torch.Tensor,
    *,
    n_qubits: int,
    ansatz_layers: int = 1,
    unified_ry_rx: bool = False,
    cnot_topology: str = "linear",
) -> torch.Tensor:
    """Batched brickwall circuit-angle θ → dense normalised ``|ψ⟩``."""
    th = theta.to(dtype=torch.float32)
    B = th.shape[0]
    n = int(n_qubits)
    layers = _brickwall_angle_layers_torch(
        th, n, ansatz_layers=ansatz_layers, unified_ry_rx=unified_ry_rx
    )
    psi = torch.ones((B, 1), device=th.device, dtype=torch.complex64)
    cnot_pairs = _cnot_pairs_for_topology(n, cnot_topology)
    for theta_y, theta_x in layers:
        cy = torch.cos(0.5 * theta_y)
        sy = torch.sin(0.5 * theta_y)
        cx = torch.cos(0.5 * theta_x)
        sx = torch.sin(0.5 * theta_x)
        q0 = torch.complex(cx * cy, -sx * sy)
        q1 = torch.complex(cx * sy, -sx * cy)
        for q in range(n):
            local = torch.stack([q0[:, q], q1[:, q]], dim=1).to(dtype=torch.complex64)
            psi = torch.einsum("bp,bs->bps", psi, local).reshape(B, -1)
        for c, t in cnot_pairs:
            perm = _cnot_permutation_torch(n, c, t, th.device)
            psi = psi[:, perm]
    return psi


def _theta_batch_to_mps_brickwall_angles_torch(
    theta: torch.Tensor,
    *,
    n_qubits: int,
    bond_dim: int,
    ansatz_layers: int = 1,
    unified_ry_rx: bool = False,
    cnot_topology: str = "linear",
) -> list[torch.Tensor]:
    """Batched brickwall circuit-angle θ → MPS tensors ``(B,L,R,2)``."""
    th = theta.to(dtype=torch.float32)
    B = th.shape[0]
    n = int(n_qubits)
    chi = max(1, int(bond_dim))
    layers = _brickwall_angle_layers_torch(
        th, n, ansatz_layers=ansatz_layers, unified_ry_rx=unified_ry_rx
    )
    tensors: list[torch.Tensor] | None = None
    for theta_y, theta_x in layers:
        if tensors is None:
            cy = torch.cos(0.5 * theta_y)
            sy = torch.sin(0.5 * theta_y)
            cx = torch.cos(0.5 * theta_x)
            sx = torch.sin(0.5 * theta_x)
            q0 = torch.complex(cx * cy, -sx * sy).to(torch.complex64)
            q1 = torch.complex(cx * sy, -sx * cy).to(torch.complex64)
            tensors = []
            for q in range(n):
                ak = torch.zeros((B, 1, 1, 2), device=th.device, dtype=torch.complex64)
                ak[:, 0, 0, 0] = q0[:, q]
                ak[:, 0, 0, 1] = q1[:, q]
                tensors.append(ak)
        else:
            gate = _rx_ry_gate_torch(theta_y, theta_x)
            for q in range(n):
                tensors[q] = torch.einsum(
                    "b l r s, b s t -> b l r t", tensors[q], gate[:, q]
                )
        tensors = _apply_brickwall_cnots_mps_torch(
            tensors, n_qubits=n, bond_dim=chi, cnot_topology=cnot_topology
        )
    assert tensors is not None
    return tensors


def _local_pauli_ops_torch(pauli_strings: list[str], n_qubits: int, device: torch.device) -> torch.Tensor:
    """Stack local Pauli operators for each string and each qubit.

    Returns tensor of shape (n_pauli, n_qubits, 2, 2) complex64.
    """
    I = torch.tensor([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]], device=device, dtype=torch.complex64)
    X = torch.tensor([[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]], device=device, dtype=torch.complex64)
    Y = torch.tensor([[0.0 + 0.0j, -1.0j], [1.0j, 0.0 + 0.0j]], device=device, dtype=torch.complex64)
    Z = torch.tensor([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -1.0 + 0.0j]], device=device, dtype=torch.complex64)
    mp = {"I": I, "X": X, "Y": Y, "Z": Z}

    p = len(pauli_strings)
    ops = torch.empty((p, int(n_qubits), 2, 2), device=device, dtype=torch.complex64)
    for pi, s in enumerate(pauli_strings):
        if len(s) != int(n_qubits):
            raise ValueError(f"Pauli string length mismatch: expected {n_qubits}, got {len(s)} for {s!r}")
        for q, ch in enumerate(s):
            ops[pi, q] = mp[ch]
    return ops


def _simulate_tn_contraction_pure_gpu(
    theta: torch.Tensor,
    *,
    pauli_strings: list[str],
    shots: int,
    seed: int,
    n_qubits: int,
    bond_dim: int,
    brickwall_ansatz_layers: int = 1,
    brickwall_unified_ry_rx: bool = False,
    brickwall_cnot_topology: str = "linear",
) -> torch.Tensor:
    """Pauli expectations ⟨P⟩ of brickwall-angle states by MPS contraction, vectorised over the Pauli index.

    Returns (B, n_pauli) float32 on the device of ``theta``.
    """
    device = theta.device
    B = theta.shape[0]
    D = int(bond_dim)
    n = int(n_qubits)
    p = len(pauli_strings)

    A_list = _theta_batch_to_mps_brickwall_angles_torch(
        theta,
        n_qubits=n,
        bond_dim=D,
        ansatz_layers=brickwall_ansatz_layers,
        unified_ry_rx=brickwall_unified_ry_rx,
        cnot_topology=brickwall_cnot_topology,
    )  # list of (B,L,R,2)
    local_ops = _local_pauli_ops_torch(pauli_strings, n_qubits=n, device=device)  # (p,n,2,2)

    # denom[b] = <psi|psi> for unnormalised MPS.
    env = torch.ones((B, 1, 1), device=device, dtype=torch.complex64)  # (b,l,m)
    I = torch.tensor([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]], device=device, dtype=torch.complex64)
    for k in range(n):
        Ak = A_list[k]  # (b,l,r,s)
        Ac = Ak.conj()
        # env_new[b,r,u] = Σ_{l,l2,s,t} env[b,l,l2] * conj(A[b,l,r,s]) * I[s,t] * A[b,l2,u,t]
        env = torch.einsum("b l m, b l r s, s t, b m u t -> b r u", env, Ac, I, Ak)
    denom = torch.real(env[:, 0, 0]).clamp(min=1e-15)  # (B,)

    # Numerator for all Pauli strings at once.
    env_p = torch.ones((B, p, 1, 1), device=device, dtype=torch.complex64)  # (b,u,l,m)
    for k in range(n):
        Ak = A_list[k]  # (b,l,r,t)
        Ac = Ak.conj()
        pk = local_ops[:, k]  # (u,s,t)
        # env_p_new[b,u,r,u2] = Σ_{l,l2,s,t} env_p[b,u,l,l2] * conj(A[b,l,r,s]) * pk[u,s,t] * A[b,l2,u2,t]
        env_p = torch.einsum("b u l m, b l r s, u s t, b m v t -> b u r v", env_p, Ac, pk, Ak)

    num = torch.real(env_p[:, :, 0, 0])  # (B,p)
    exp = (num / denom[:, None]).clamp(-1.0, 1.0)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return _sample_pauli_expectation_torch(exp, shots, generator=gen)


def _local_pauli_ops_numpy(pauli_strings: list[str], n_qubits: int) -> np.ndarray:
    """(p,n,2,2) complex128 local Pauli ops on CPU."""
    I = np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]], dtype=np.complex128)
    X = np.array([[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]], dtype=np.complex128)
    Y = np.array([[0.0 + 0.0j, -1.0j], [1.0j, 0.0 + 0.0j]], dtype=np.complex128)
    Z = np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -1.0 + 0.0j]], dtype=np.complex128)
    mp = {"I": I, "X": X, "Y": Y, "Z": Z}

    p = len(pauli_strings)
    ops = np.empty((p, int(n_qubits), 2, 2), dtype=np.complex128)
    for pi, s in enumerate(pauli_strings):
        if len(s) != int(n_qubits):
            raise ValueError(f"Pauli string length mismatch: expected {n_qubits}, got {len(s)} for {s!r}")
        for q, ch in enumerate(s):
            ops[pi, q] = mp[ch]
    return ops


def _simulate_tn_contraction_pure_cpu(
    theta_np: np.ndarray,
    *,
    pauli_strings: list[str],
    shots: int,
    seed: int,
    n_qubits: int,
    bond_dim: int,
    brickwall_ansatz_layers: int = 1,
    brickwall_unified_ry_rx: bool = False,
    brickwall_cnot_topology: str = "linear",
) -> np.ndarray:
    """CPU MPS contraction of brickwall-angle states (no dense |ψ⟩)."""
    theta_np = np.asarray(theta_np, dtype=np.float64)
    N = theta_np.shape[0]
    n_pauli = len(pauli_strings)
    n = int(n_qubits)
    D = int(bond_dim)
    ops_local = _local_pauli_ops_numpy(pauli_strings, n_qubits=n)  # (p,n,2,2)
    I = np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]], dtype=np.complex128)
    rng = np.random.default_rng(seed)
    out = np.empty((N, n_pauli), dtype=np.float32)

    for i in range(N):
        A_list = theta_to_mps_tensors(
            theta_np[i],
            n_qubits=n,
            bond_dim=D,
            ansatz_layers=brickwall_ansatz_layers,
            unified_ry_rx=brickwall_unified_ry_rx,
            cnot_topology=brickwall_cnot_topology,
        )  # per-site (L,R,s)

        # denom = <psi|psi> for unnormalised MPS
        env = np.ones((1, 1), dtype=np.complex128)
        for k in range(n):
            Ak = A_list[k]  # (L,R,s)
            # env_new[c,d] = Σ_{a,b,s,t} env[a,b] * conj(A[a,c,s]) * I[s,t] * A[b,d,t]
            env = np.einsum("a b, a c s, s t, b d t -> c d", env, np.conjugate(Ak), I, Ak)
        denom = float(np.real(env[0, 0]))
        denom = max(denom, 1e-15)

        exps = np.empty((n_pauli,), dtype=np.float64)
        for pi in range(n_pauli):
            env_p = np.ones((1, 1), dtype=np.complex128)
            for k in range(n):
                Ak = A_list[k]
                pk = ops_local[pi, k]
                env_p = np.einsum(
                    "a b, a c s, s t, b d t -> c d",
                    env_p,
                    np.conjugate(Ak),
                    pk,
                    Ak,
                )
            num = float(np.real(env_p[0, 0]))
            exps[pi] = num / denom

        exps = np.clip(exps, -1.0, 1.0)
        out[i] = _sample_pauli_expectation_numpy(rng, exps, shots)
    return out


def _simulate_numpy_vectorized(
    theta_np: np.ndarray,
    P_stack_np: np.ndarray,
    shots: int,
    seed: int,
    chunk_size: int = 20_000,
) -> np.ndarray:
    """Vectorised NumPy simulation (BLAS matmul, no Python loops). Returns (N, n_pauli) float32."""
    rng = np.random.default_rng(seed)
    P = P_stack_np.astype(np.complex128)  # (n_pauli, d, d)
    n_pauli = P.shape[0]
    N = len(theta_np)
    out = np.empty((N, n_pauli), dtype=np.float32)

    for start in tqdm(range(0, N, chunk_size), desc="NumPy sim", leave=False):
        end = min(start + chunk_size, N)
        psi = _theta_batch_to_psi(theta_np[start:end])   # (B, d) complex128
        # P[p,d,j] · ψ[b,j] → Ppsi[b,p,d]
        # np.matmul(P, psi.T): (n_pauli,d,d)@(d,B) → (n_pauli,d,B)
        Ppsi_T = np.matmul(P, psi.T)                      # (n_pauli, d, B)
        # exp[b,p] = Re(Σ_d ψ*[b,d] · Ppsi_T[p,d,b])
        exp = np.real(
            np.einsum("bd,pdb->bp", psi.conj(), Ppsi_T)
        ).clip(-1.0, 1.0)                                  # (B, n_pauli)
        out[start:end] = _sample_pauli_expectation_numpy(rng, exp, shots)

    return out


def _simulate_gpu_circuit(
    theta: torch.Tensor,
    P_stack_np: np.ndarray | None,
    shots: int | None,
    seed: int,
    *,
    n_qubits: int,
    tensor_bond_dim: int,
    pauli_strings: list[str],
    tn_avoid_dense: bool = False,
    brickwall_ansatz_layers: int = 1,
    brickwall_unified_ry_rx: bool = False,
    brickwall_cnot_topology: str = "linear",
    chunk_size: int = 10_000,
) -> np.ndarray:
    """CUDA simulation of brickwall-angle states (dense |ψ⟩ or MPS contraction)."""
    device = torch.device("cuda")
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    n_pauli = len(pauli_strings)
    P_stack: torch.Tensor | None = None
    if P_stack_np is not None:
        P_stack = torch.as_tensor(P_stack_np, dtype=torch.complex64, device=device)  # (p,d,d)

    n = int(n_qubits)
    chunk_size = min(chunk_size, 1_000 if tn_avoid_dense else 50_000)

    N = theta.shape[0]
    out_gpu = torch.empty((N, n_pauli), dtype=torch.float32, device=device)

    for start in tqdm(range(0, N, chunk_size), desc="GPU sim (general)", leave=False):
        end = min(start + chunk_size, N)
        chunk = theta[start:end].to(dtype=torch.float32, device=device)  # (B,num_params)

        if tn_avoid_dense:
            out_chunk = _simulate_tn_contraction_pure_gpu(
                chunk,
                pauli_strings=pauli_strings,
                shots=shots,
                seed=seed + start,
                n_qubits=n,
                bond_dim=int(tensor_bond_dim),
                brickwall_ansatz_layers=brickwall_ansatz_layers,
                brickwall_unified_ry_rx=brickwall_unified_ry_rx,
                brickwall_cnot_topology=brickwall_cnot_topology,
            ).to(torch.float32)
            out_gpu[start:end] = out_chunk.to(device=device)
            continue

        psi = _theta_batch_to_psi_brickwall_angles_torch(
            chunk,
            n_qubits=n,
            ansatz_layers=brickwall_ansatz_layers,
            unified_ry_rx=brickwall_unified_ry_rx,
            cnot_topology=brickwall_cnot_topology,
        )
        Ppsi = torch.einsum("pdj,bj->bpd", P_stack, psi)  # (B,n_pauli,d)
        exp = torch.real(torch.einsum("bd,bpd->bp", psi.conj(), Ppsi)).clamp(-1.0, 1.0)
        out_gpu[start:end] = _sample_pauli_expectation_torch(exp, shots, generator=gen)

    n_out, p_out = out_gpu.shape
    print(
        f"  GPU sim done — transferring x to CPU ({n_out}×{p_out} float32, "
        f"~{n_out * p_out * 4 / 1e6:.0f} MB)..."
    )
    return out_gpu.cpu().numpy().astype(np.float32)


def _simulate_dense_cpu(
    theta_np: np.ndarray,
    P_stack: np.ndarray,
    shots: int | None,
    seed: int,
    *,
    n_qubits: int,
    brickwall_ansatz_layers: int,
    brickwall_unified_ry_rx: bool,
    brickwall_cnot_topology: str,
) -> np.ndarray:
    """CPU simulation of brickwall-angle states through the dense |ψ⟩ (row by row)."""
    rng = np.random.default_rng(seed)
    P = P_stack.astype(np.complex128)  # (p,d,d)
    n_pauli = P.shape[0]
    N = len(theta_np)
    out = np.empty((N, n_pauli), dtype=np.float32)
    for i in tqdm(range(N), desc="CPU sim (circuit_angles)", leave=False):
        psi = theta_to_psi_brickwall_angles(
            theta_np[i],
            n_qubits=n_qubits,
            ansatz_layers=brickwall_ansatz_layers,
            unified_ry_rx=brickwall_unified_ry_rx,
            cnot_topology=brickwall_cnot_topology,
        )
        exps = np.empty(n_pauli, dtype=np.float64)
        for pi in range(n_pauli):
            exps[pi] = expectation_pauli_pure(psi, P[pi])
        out[i] = _sample_pauli_expectation_numpy(rng, np.clip(exps, -1.0, 1.0), shots)
    return out


def simulate_parallel(
    theta: "np.ndarray | torch.Tensor",
    pauli_strings: list[str],
    shots: int | None,
    seed: int,
    desc: str = "QST sim",
    *,
    state_param: str = "circuit_angles",
    n_qubits: int | None = None,
    tensor_bond_dim: int = 2,
    tn_avoid_dense: bool = False,
    brickwall_ansatz_layers: int = 1,
    brickwall_unified_ry_rx: bool = False,
    brickwall_cnot_topology: str = "linear",
) -> np.ndarray:
    """Pauli expectations ⟨P⟩ ∈ [-1, 1] for every row of ``theta`` (shape (N, n_pauli), float32).

    ``theta`` may be a NumPy array or a torch tensor (CPU or CUDA; a CUDA tensor avoids an H2D copy).
    ``shots=None`` gives exact expectations, an int gives binomial shot noise. Runs on CUDA when
    available (cholesky falls back to NumPy on out-of-memory), otherwise on CPU. ``tn_avoid_dense``
    selects the MPS contraction (bond dimension ``tensor_bond_dim``) instead of the dense 2ⁿ state.
    """
    if state_param not in ("cholesky", "circuit_angles"):
        raise ValueError(f"Unsupported state_param={state_param!r}; use 'cholesky' or 'circuit_angles'")
    n = theta.shape[0]
    theta_np = theta.cpu().numpy() if isinstance(theta, torch.Tensor) else theta
    n_pauli = len(pauli_strings)
    cholesky = state_param == "cholesky"
    if not cholesky and n_qubits is None:
        raise ValueError("simulate_parallel: n_qubits must be provided for circuit_angles")

    # Dense 2ⁿ Pauli matrices cost O(4ⁿ) memory/time; the MPS contraction only needs local 2×2 operators.
    P_stack = None
    if cholesky or not tn_avoid_dense:
        print(f"  [{desc}] Building dense Pauli stack ({n_pauli} strings, d=2^n)...")
        P_stack = _build_pauli_stack(pauli_strings)  # (n_pauli, d, d) complex64

    if torch.cuda.is_available():
        print(
            f"  [{desc}] GPU vectorised: N={n}, n_pauli={n_pauli}, shots={shots}"
            f"  (device: {torch.cuda.get_device_name(0)})"
        )
        if cholesky:
            try:
                return _simulate_gpu(theta, P_stack, shots, seed)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                warnings.warn(
                    f"[{desc}] CUDA OOM during simulation — falling back to NumPy path.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        else:
            theta_t = theta if isinstance(theta, torch.Tensor) else torch.as_tensor(
                theta, dtype=torch.float32, device="cuda"
            )
            return _simulate_gpu_circuit(
                theta_t,
                P_stack,
                shots,
                seed,
                n_qubits=n_qubits,
                tensor_bond_dim=tensor_bond_dim,
                pauli_strings=pauli_strings,
                tn_avoid_dense=tn_avoid_dense,
                brickwall_ansatz_layers=brickwall_ansatz_layers,
                brickwall_unified_ry_rx=brickwall_unified_ry_rx,
                brickwall_cnot_topology=brickwall_cnot_topology,
            )

    print(f"  [{desc}] NumPy vectorised: N={n}, n_pauli={n_pauli}, shots={shots}")
    if cholesky:
        return _simulate_numpy_vectorized(theta_np, P_stack, shots, seed)
    if tn_avoid_dense:
        print(f"  [{desc}] CPU MPS transfer-matrix contraction")
        return _simulate_tn_contraction_pure_cpu(
            theta_np,
            pauli_strings=pauli_strings,
            shots=shots,
            seed=seed,
            n_qubits=n_qubits,
            bond_dim=tensor_bond_dim,
            brickwall_ansatz_layers=brickwall_ansatz_layers,
            brickwall_unified_ry_rx=brickwall_unified_ry_rx,
            brickwall_cnot_topology=brickwall_cnot_topology,
        )
    return _simulate_dense_cpu(
        theta_np,
        P_stack,
        shots,
        seed,
        n_qubits=n_qubits,
        brickwall_ansatz_layers=brickwall_ansatz_layers,
        brickwall_unified_ry_rx=brickwall_unified_ry_rx,
        brickwall_cnot_topology=brickwall_cnot_topology,
    )
