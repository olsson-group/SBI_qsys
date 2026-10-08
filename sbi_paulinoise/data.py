"""Data loading and inline simulation for SNPE."""

import csv
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sbi_paulinoise.paulinoise_simulator_2 import NoisyCircuitSimulator as _Sim2, load_init_states_prep_blochs as _load_prep_blochs


# ── Observables CSV ───────────────────────────────────────────────────────────

def load_observables(path: str = "paulinoise_observables.csv") -> list[str]:
    """Return list of observable strings (skipping header row)."""
    with open(path) as f:
        return [row[0] for row in csv.reader(f) if row and row[0] != "Observable String"]


# ── Simulator dimensions ──────────────────────────────────────────────────────

def resolve_init_prep_blochs(config: dict) -> tuple[list | None, int]:
    """Return (blochs, n_states) for the configured initial states.

    INIT_STATES_FILE set  → load from CSV.
    INIT_STATES_FILE None → None (single default Ry·Rz|0⟩ prep, used by non-subset mode).
                            OBS_SUBSET_K mode samples from 4^n states internally.
    """
    nq = config["N_QUBITS"]
    path = config.get("INIT_STATES_FILE")
    if not path:
        return None, 1
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"INIT_STATES_FILE={path!r} not found. Generate it with paulinoise_simulator_2.py."
        )
    blochs = _load_prep_blochs(path, nq)
    return blochs, len(blochs)


def get_dims(config: dict, observables: list) -> tuple[int, int]:
    """Return (num_params, n_obs) for the given config.

    OBS_SUBSET_K set (obs_learning): n_obs = obs_k * 3.
    Each slot encodes [state_idx/(4^n-1), pauli_idx/(n_obs-1), val].
    Otherwise: n_obs = len(observables) * n_states (all states × all Paulis).
    """
    nq = config["N_QUBITS"]
    tmp = _Sim2(n_qubits=nq, cx_topology=config["CX_TOPOLOGY"])
    num_params = tmp.total_params
    del tmp
    _, n_states = resolve_init_prep_blochs(config)
    n_obs = len(observables) * n_states
    return num_params, n_obs


def _bloch_to_ry_rz(bloch: np.ndarray) -> np.ndarray:
    """Convert per-qubit Bloch vectors (n_qubits, 3) to flat Ry/Rz angles for sim1.

    sim1 prep is Rz(rz)·Ry(ry)|0⟩, giving Bloch vector:
        bx = sin(ry)cos(rz),  by = sin(ry)sin(rz),  bz = cos(ry)
    Inverse: ry = arccos(bz),  rz = arctan2(by, bx).
    Returns flat array [ry0, rz0, ry1, rz1, ...] of length 2*n_qubits.
    """
    bx, by, bz = bloch[:, 0], bloch[:, 1], bloch[:, 2]
    ry = np.arccos(np.clip(bz, -1.0, 1.0))
    rz = np.arctan2(by, bx)
    return np.stack([ry, rz], axis=1).ravel().astype(np.float64)


def _compute_xobs_sim2_full(
    theta_truth: np.ndarray,
    n_qubits: int,
    observables: list,
    cx_topology: str,
    init_prep_blochs: list | None,
) -> np.ndarray:
    """Full-mode x_obs via paulinoise_simulator_2 (analytic fallback)."""
    theta_row = np.asarray(theta_truth).ravel()
    sim2 = _Sim2(n_qubits=n_qubits, cx_topology=cx_topology)
    if init_prep_blochs is None:
        vals = np.array(sim2.compute_expectations(theta_row, observables), dtype=np.float32)
        return vals.reshape(1, -1)
    parts = [np.array(sim2.compute_expectations(theta_row, observables, prep_bloch=b),
                      dtype=np.float32) for b in init_prep_blochs]
    return np.concatenate(parts).reshape(1, -1)


def _compute_xobs_sim1_full(
    theta_truth: np.ndarray,
    n_qubits: int,
    observables: list,
    cx_topology: str,
    init_prep_blochs: list | None,
) -> np.ndarray:
    """Compute full-mode x_obs using paulinoise_simulator_1 (Aer density matrix).

    Mirrors the full-mode simulation in _sim_row: runs all init states × all
    observables and concatenates, matching the n_obs layout expected by the flow.
    Returns shape (1, n_obs).
    """
    try:
        from sbi_paulinoise.paulinoise_simulator_1 import NoisyCircuitSimulator as _Sim1
    except (ImportError, OSError) as e:
        print(f"  [Warning] paulinoise_simulator_1 unavailable ({e}); falling back to sim2 for x_obs.")
        return _compute_xobs_sim2_full(theta_truth, n_qubits, observables, cx_topology,
                                       init_prep_blochs)
    theta_row = np.asarray(theta_truth).ravel()
    sim1 = _Sim1(n_qubits=n_qubits, cx_topology=cx_topology)
    if init_prep_blochs is None:
        vals = np.array(sim1.compute_expectations(theta_row, observables, method='aer_dm'),
                        dtype=np.float32)
        return vals.reshape(1, -1)
    parts = []
    for bloch in init_prep_blochs:
        prep_flat = _bloch_to_ry_rz(bloch)
        vals = np.array(
            sim1.compute_expectations(theta_row, observables, method='aer_dm',
                                      prep_ry_rz_flat=prep_flat),
            dtype=np.float32,
        )
        parts.append(vals)
    return np.concatenate(parts).reshape(1, -1)


def load_snpe(config: dict) -> dict:
    """Load observed x and optional ground truth for SNPE.

    Two modes:
      OBS_SUBSET_K=None : full mode — INIT_STATES_FILE required,
                          obs_dim = n_states × n_obs.
      OBS_SUBSET_K=k    : subset mode — per-qubit states sampled from 4^n × n_obs,
                          obs_dim = k × (n_qubits + 2). INIT_STATES_FILE not required.

    Returns dict: x_obs_np, theta_truth_np (may be None),
                  num_params, n_obs, plot_ground_truth, observables
    """
    obs_k = config.get("OBS_SUBSET_K")
    if obs_k is None and not config.get("INIT_STATES_FILE"):
        raise ValueError(
            "OBS_SUBSET_K=None (full mode) requires INIT_STATES_FILE to be set."
        )
    observables = load_observables(config.get("OBSERVABLES_FILE", "paulinoise_observables.csv"))
    num_params, n_obs = get_dims(config, observables)
    nq = config["N_QUBITS"]
    print(f"  SNPE simulator: {nq} qubits, {num_params} params, "
          f"obs_dim={n_obs} ({config.get('LEARNING_MODE', 'obs_learning')})")

    # ── Load theta_truth (required for subset mode, optional for full mode) ───
    truth_file = config.get("THETA_TRUTH_FILE", "paulinoise_1_parameters.csv")
    try:
        theta_truth_np = np.loadtxt(truth_file, delimiter=",")
        if theta_truth_np.ndim == 1:
            theta_truth_np = theta_truth_np.reshape(1, -1)
    except OSError:
        theta_truth_np = None
        if obs_k is not None:
            raise RuntimeError(
                f"OBS_SUBSET_K mode requires THETA_TRUTH_FILE but {truth_file} not found."
            )
        print(f"  {truth_file} not found.")

    init_prep_blochs_tmp, _ = resolve_init_prep_blochs(config)

    if theta_truth_np is not None:
        x_obs_np = _compute_xobs_sim1_full(
            theta_truth_np, nq, observables,
            config["CX_TOPOLOGY"],
            init_prep_blochs_tmp,
        )
        print(f"  x_obs simulated from theta_truth via sim1/aer_dm ({x_obs_np.shape[1]} dims)")
    else:
        x_obs_file = config.get("X_OBS_FILE")
        if not x_obs_file:
            raise ValueError("THETA_TRUTH_FILE not found and X_OBS_FILE not set.")
        try:
            x_obs_np = np.loadtxt(x_obs_file, delimiter=",")
        except OSError as e:
            raise RuntimeError(f"Error loading {x_obs_file}: {e}") from e
        if x_obs_np.ndim == 0:
            x_obs_np = x_obs_np.reshape(1, 1)
        elif x_obs_np.ndim == 1:
            x_obs_np = x_obs_np.reshape(1, -1)
        if x_obs_np.shape[-1] != n_obs:
            raise ValueError(
                f"Observation dimension mismatch: {x_obs_file} has {x_obs_np.shape[-1]} columns, "
                f"but config expects n_obs={n_obs}."
            )

    return dict(
        x_obs_np=x_obs_np,
        theta_truth_np=theta_truth_np,
        num_params=num_params,
        n_obs=n_obs,
        observables=observables,
        init_prep_blochs=init_prep_blochs_tmp,
    )
