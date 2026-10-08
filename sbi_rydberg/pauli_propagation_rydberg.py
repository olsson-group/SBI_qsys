"""Batched Pauli back-propagation (Heisenberg picture) for Rydberg lattices.

H = Ω Σ_i X_i + Δ Σ_i Z_i + Σ_{i<j} V_ij Z_i Z_j,   V_ij = scale · C6 / r_ij^6

One first-order Trotter step is U = Rx_all · Rz_all · RZZ_all. Each observable
(Z_i, and Z_i Z_j on NN/NNN bonds) is conjugated through U step by step, with
small Pauli terms truncated, and evaluated on |+...+⟩. The batch dimension N
(different atom displacements θ) lives in the coefficient tensor.

State: paulis (n_terms, n_atoms) int8 [I=0, X=1, Y=2, Z=3]; coeffs (n_terms, N) float32.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from sbi_rydberg.lattice_simulate import (
    RydbergSimConfig,
    ideal_square_positions,
    nearest_neighbor_pairs,
    next_nearest_neighbor_pairs,
)

# Z·P = (sign) · P' for P in (I, X, Y, Z); {X, Y} anticommute with Z.
_ZP_NEW = [3, 2, 1, 0]
_ZP_SIGN = [1.0, -1.0, 1.0, 1.0]
_ANTI_Z = [False, True, True, False]


def _apply_rx(paulis, coeffs, k, cos_a, sin_a):
    col = paulis[:, k]
    y, z = col == 2, col == 3
    c_y, c_z = coeffs[y], coeffs[z]
    p_y, p_z = paulis[y].clone(), paulis[z].clone()
    p_y[:, k] = 3
    p_z[:, k] = 2
    coeffs[y] = c_y * cos_a
    coeffs[z] = c_z * cos_a
    return torch.cat([paulis, p_y, p_z]), torch.cat([coeffs, -sin_a * c_y, sin_a * c_z])


def _apply_rz(paulis, coeffs, k, cos_b, sin_b):
    col = paulis[:, k]
    x, y = col == 1, col == 2
    c_x, c_y = coeffs[x], coeffs[y]
    p_x, p_y = paulis[x].clone(), paulis[y].clone()
    p_x[:, k] = 2
    p_y[:, k] = 1
    coeffs[x] = c_x * cos_b
    coeffs[y] = c_y * cos_b
    return torch.cat([paulis, p_x, p_y]), torch.cat([coeffs, -sin_b * c_x, sin_b * c_y])


def _apply_rzz(paulis, coeffs, qi, qj, gamma, tables):
    """gamma: (N,) rotation angles per sample."""
    anti_z, zp_new, zp_sign = tables
    ci, cj = paulis[:, qi].long(), paulis[:, qj].long()
    anti = anti_z[ci] ^ anti_z[cj]
    anti_c = coeffs[anti].clone()
    if anti_c.shape[0] == 0:
        return paulis, coeffs
    coeffs[anti] = anti_c * torch.cos(gamma)
    new_p = paulis[anti].clone()
    new_p[:, qi] = zp_new[ci[anti]]
    new_p[:, qj] = zp_new[cj[anti]]
    sign = zp_sign[ci[anti]] * zp_sign[cj[anti]]
    new_c = sign[:, None] * torch.sin(gamma)[None, :] * anti_c
    return torch.cat([paulis, new_p]), torch.cat([coeffs, new_c])


def _merge(paulis, coeffs):
    """Sum coefficients of identical Pauli strings (sort + scatter_add)."""
    n, n_atoms = paulis.shape
    if n <= 1:
        return paulis, coeffs

    # Base-4 int64 keys, 31 qubits per key (4^31 < 2^63); stable multi-pass lexsort.
    chunk = 31
    keys = []
    for c in range(0, n_atoms, chunk):
        sl = paulis[:, c:c + chunk].to(torch.int64)
        pw = 4 ** torch.arange(sl.shape[1], device=paulis.device, dtype=torch.int64)
        keys.append((sl * pw).sum(dim=1))
    idx = torch.arange(n, device=paulis.device)
    for key in reversed(keys):
        idx = idx[torch.argsort(key[idx], stable=True)]

    keys_s = torch.stack([k[idx] for k in keys], dim=1)
    boundary = torch.ones(n, dtype=torch.bool, device=paulis.device)
    boundary[1:] = (keys_s[1:] != keys_s[:-1]).any(dim=1)
    group = boundary.long().cumsum(0) - 1
    out_c = torch.zeros(int(group[-1]) + 1, coeffs.shape[1], device=coeffs.device, dtype=coeffs.dtype)
    out_c.scatter_add_(0, group.unsqueeze(1).expand(-1, coeffs.shape[1]), coeffs[idx])
    return paulis[idx][boundary], out_c


def _truncate(paulis, coeffs, eps, max_weight):
    keep = coeffs.abs().amax(dim=1) >= eps
    if max_weight is not None:
        keep &= (paulis != 0).sum(dim=1) <= max_weight
    return paulis[keep], coeffs[keep]


def _expectation_plus(paulis, coeffs):
    """⟨+...+|O|+...+⟩ = sum of coefficients of all-I/X strings, shape (N,)."""
    ix = ((paulis == 0) | (paulis == 1)).all(dim=1)
    return coeffs[ix].sum(dim=0)


def _build_interactions(theta, ideal_pos, cfg):
    """Pairs active (|V| ≥ v_threshold) for any sample in the batch, and V of shape (N, n_pairs)."""
    N, n_atoms = theta.shape[0], ideal_pos.shape[0]
    pos = ideal_pos[None] + theta.reshape(N, n_atoms, 2)
    pairs, V = [], []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            r = np.sqrt(np.maximum(((pos[:, i] - pos[:, j]) ** 2).sum(axis=1), 1e-12))
            v = cfg.interaction_scale * cfg.c6_rad_um6_per_us / r ** 6
            if np.max(np.abs(v)) >= cfg.v_threshold:
                pairs.append((i, j))
                V.append(v)
    return pairs, (np.stack(V, axis=1) if V else np.empty((N, 0)))


def _resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    return device


def _simulate_chunk(theta, cfg, eps, max_weight, device, observables):
    N, n_atoms = theta.shape[0], cfg.n_atoms
    ideal_pos = ideal_square_positions(cfg.nx, cfg.ny, cfg.lattice_constant_um)

    if observables is None:
        bonds = nearest_neighbor_pairs(cfg.nx, cfg.ny) + next_nearest_neighbor_pairs(cfg.nx, cfg.ny)
        obs_init = []
        for sites in [(i,) for i in range(n_atoms)] + bonds:
            row = np.zeros(n_atoms, dtype=np.int8)
            row[list(sites)] = 3
            obs_init.append(row)
    else:
        obs_init = [np.asarray(o, dtype=np.int8).reshape(-1) for o in observables]
    n_per_t = len(obs_init)

    pairs, V = _build_interactions(theta, ideal_pos, cfg)
    dt, Omega, Delta = cfg.dt_us, cfg.omega_rad_per_us, cfg.detuning_rad_per_us
    gammas = [torch.as_tensor(2.0 * dt * V[:, p].astype(np.float32), device=device) for p in range(len(pairs))]
    tables = (
        torch.tensor(_ANTI_Z, device=device, dtype=torch.bool),
        torch.tensor(_ZP_NEW, device=device, dtype=torch.int8),
        torch.tensor(_ZP_SIGN, device=device, dtype=torch.float32),
    )

    snap_steps = [round(t / dt) for t in cfg.times_us]
    cos_rx, sin_rx = float(np.cos(2 * Omega * dt)), float(np.sin(2 * Omega * dt))
    cos_rz, sin_rz = float(np.cos(2 * Delta * dt)), float(np.sin(2 * Delta * dt))
    do_rz = abs(Delta) > 1e-15

    def merge_truncate(p, c):
        return _truncate(*_merge(p, c), eps, max_weight)

    result = np.zeros((N, n_per_t * len(snap_steps)), dtype=np.float32)
    for obs_idx, row in enumerate(obs_init):
        paulis = torch.as_tensor(row[None, :], device=device)
        coeffs = torch.ones((1, N), dtype=torch.float32, device=device)
        for step in range(1, max(snap_steps) + 1):
            # Conjugation order is the reverse of U: Rx -> Rz -> RZZ.
            for k in range(n_atoms):
                paulis, coeffs = _apply_rx(paulis, coeffs, k, cos_rx, sin_rx)
            paulis, coeffs = merge_truncate(paulis, coeffs)
            if do_rz:
                for k in range(n_atoms):
                    paulis, coeffs = _apply_rz(paulis, coeffs, k, cos_rz, sin_rz)
                paulis, coeffs = merge_truncate(paulis, coeffs)
            for p, (qi, qj) in enumerate(pairs):
                paulis, coeffs = _apply_rzz(paulis, coeffs, qi, qj, gammas[p], tables)
                # Merge/truncate every 4 RZZ gates (not every gate) to bound term growth.
                if (p + 1) % 4 == 0 or p == len(pairs) - 1:
                    paulis, coeffs = merge_truncate(paulis, coeffs)
            for t_idx, s in enumerate(snap_steps):
                if s == step:
                    result[:, t_idx * n_per_t + obs_idx] = _expectation_plus(paulis, coeffs).cpu().numpy()
    return result


def simulate_batch_pauli_propagation(
    theta: np.ndarray,
    cfg: RydbergSimConfig,
    *,
    eps: float = 1e-6,
    max_weight: int | None = None,
    observables: list[np.ndarray] | None = None,
    device: str = "cpu",
    chunk_size: int = 512,
    verbose: bool = False,
) -> np.ndarray:
    """θ (N, 2*n_atoms) displacements in μm -> x (N, cfg.n_obs).

    Columns are time-major: [Z_i..., ZZ_NN..., ZZ_NNN...] per snapshot time, or one column per
    entry of ``observables`` (Pauli rows, I=0 X=1 Y=2 Z=3, length n_atoms) if given.
    eps drops terms with max_N |coeff| < eps; max_weight drops higher-weight strings.
    Samples are processed in chunks of ``chunk_size`` to bound memory.
    """
    theta = np.asarray(theta, dtype=np.float64)
    device = _resolve_device(device)
    N = theta.shape[0]
    parts = []
    for ci, i in enumerate(range(0, N, chunk_size)):
        t0 = time.perf_counter()
        parts.append(_simulate_chunk(theta[i:i + chunk_size], cfg, eps, max_weight, device, observables))
        if verbose:
            print(f"  [PP] chunk {ci + 1}/{-(-N // chunk_size)}  {time.perf_counter() - t0:.1f}s", flush=True)
    return np.concatenate(parts, axis=0)
