"""Amortised quantum state tomography with NPE: Pauli expectations → posterior over the state parameters.

Two configurations reproduce the results in ``sbi_qst_results/``:

* default — 12-qubit brickwall circuit-angle states (``STATE_PARAM='circuit_angles'``), the NPE is
  trained on ``NPE_N_SIMS`` random circuits and evaluated on held-out random circuits
  (``qst_<time>_nsims<N>/``; ``QST_NPE_N_SIMS`` selects the simulation budget).
* ``QST_PRESET=prep_sweep_cholesky`` — 4-qubit dense (``cholesky``) states with a Gaussian prior, one NPE
  trained on a reference brickwall state and tested on a grid of prep angles (``qst_prep_sweep_<time>/``).

Run from the repository root::

    python sbi_qst/run_sbi.py
"""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
import time
from datetime import datetime

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from sbi_qst import visualize as viz
from sbi_qst.physics import (
    brickwall_rx_ry_state_vector,
    brickwall_theta_keys,
    num_params_brickwall_angles,
    pauli_strings_weight1_nn,
    prep_to_theta,
    random_pauli_strings,
    sample_brickwall_angle_theta_unique,
    theta_to_psi_brickwall_angles,
)
from sbi_qst.simulate import simulate_parallel
from sbi_qst.train_npe import (
    _PassThroughX,
    _StandardizeX,
    _build_density_estimator,
    draw_posterior_samples,
    train_npe,
)

_PRESET = os.environ.get("QST_PRESET", "random_state")

N_QUBITS = int(os.environ.get("QST_N_QUBITS", 4 if _PRESET == "prep_sweep_cholesky" else 12))
DIM = 2**N_QUBITS

# STATE_PARAM / PRIOR_TYPE pairs:
#   "circuit_angles" / "brickwall_angles"  θ = brickwall Rx/Ry angles, sampled uniformly in SWEEP_THETA_Y/X
#       BRICKWALL_PER_QUBIT_PREP + BRICKWALL_UNIFIED_RY_RX: n angles per layer (Ry(θ_q)Rx(θ_q) per qubit)
#       BRICKWALL_PER_QUBIT_PREP only: 2n (independent θ_y[q], θ_x[q]);  neither: 2 shared angles
#   "cholesky" / "gaussian"                θ = Re/Im of the dense state vector (2·2ⁿ)
CONFIG = {
    "N_QUBITS": N_QUBITS,
    "STATE_PARAM": "circuit_angles",
    "PRIOR_TYPE": "brickwall_angles",
    # MPS bond dimension of the circuit_angles forward model; 2 is exact for one layer at any n.
    "TENSOR_BOND_DIM": 2,
    # True: Pauli expectations by MPS contraction (no dense 2ⁿ state); False: dense, faster for n ≲ 5.
    "TN_AVOID_DENSE": N_QUBITS >= 6,
    # Reference brickwall state used for the conditioning x_obs (and the sweep reference).
    "PREP_THETA_Y": float(np.pi / 3.0),
    "PREP_THETA_X": float(np.pi / 3.0),
    # True: after training, test an SWEEP_N_Y×SWEEP_N_X grid of prep angles (endpoint excluded) instead
    # of the held-out random circuits.
    "PREP_ANGLE_SWEEP": False,
    "SWEEP_N_Y": 12,
    "SWEEP_N_X": 12,
    "SWEEP_THETA_Y": (-np.pi, np.pi),
    "SWEEP_THETA_X": (-np.pi, np.pi),
    # Z-score each Pauli expectation with the mean/std of the training simulations.
    "STANDARDIZE_X": True,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    # Observations are Pauli expectations ⟨P⟩ ∈ [-1, 1]; None → exact, int → binomial shot noise.
    "SHOTS": 10000,
    "BRICKWALL_PER_QUBIT_PREP": True,
    "BRICKWALL_UNIFIED_RY_RX": True,
    "BRICKWALL_ANSATZ_LAYERS": 1,
    # CNOT order within a layer: "linear" (chain) or "brickwall" (odd bonds, then even bonds).
    "BRICKWALL_CNOT_TOPOLOGY": "linear",
    # Box of sbi's bookkeeping prior when the θ width differs from the per-layer layout (multi-layer).
    "PRIOR_LOW": -3.0,
    "PRIOR_HIGH": 3.0,
    "NPE_N_SIMS": int(os.environ.get("QST_NPE_N_SIMS", 900_000)),
    "SIM_SEED": 42,
    "OBS_SEED": 123,
    # θ rounded to this many decimals when removing duplicates (and training θ from the held-out set).
    "THETA_DEDUP_DECIMALS": 12,
    "EPOCHS": 10000,
    "BATCH_SIZE": 25600,
    "NPE_DATA_DEVICE": "cuda",
    "STOP_AFTER_EPOCHS": 100,
    "LR": 1e-4,
    "CLIP_MAX_NORM": 5.0,
    "NUM_POSTERIOR_SAMPLES": 500,
    # Held-out random circuits (circuit_angles only); 0 disables. None layers → BRICKWALL_ANSATZ_LAYERS.
    "RANDOM_EVAL_N_STATES": 200,
    "RANDOM_EVAL_PER_QUBIT_PREP": True,
    "RANDOM_EVAL_UNIFIED_RY_RX": True,
    "RANDOM_EVAL_ANSATZ_LAYERS": None,
    "FLOW_MODEL": "zuko_nsf",
    "FLOW_HIDDEN_FEATURES": 2048,
    "NUM_TRANSFORMS": 3,
    "NSF_NUM_BINS": 16,
    # Observed Paulis: "weight1_nn" (weight-1 + nearest-neighbour weight-2, 12n-9 strings) or "random".
    "PAULI_OBS_MODE": "weight1_nn",
    "N_RANDOM_PAULIS": 180,
    "PAULI_DRAW_SEED": 999,
    "SAVE_SWEEP_X_OBS_GRID": True,
    "SAVE_NPE_TRAINING_NPZ": False,
    "NPE_TRAINING_NPZ_COMPRESSED": False,
    # Dense ρ heatmaps need a 2ⁿ×2ⁿ matrix: skipped above this many qubits.
    "MAX_N_QUBITS_DENSE_RHO_PLOTS": min(N_QUBITS, 12),
}

if _PRESET == "prep_sweep_cholesky":
    CONFIG.update({
        "STATE_PARAM": "cholesky",
        "PRIOR_TYPE": "gaussian",
        "TENSOR_BOND_DIM": 4,
        "TN_AVOID_DENSE": True,
        "PREP_ANGLE_SWEEP": True,
        "PRIOR_LOW": -1.0,
        "PRIOR_HIGH": 1.0,
        "BRICKWALL_PER_QUBIT_PREP": False,
        "BRICKWALL_UNIFIED_RY_RX": False,
        "NPE_N_SIMS": 1_000_000,
        "BATCH_SIZE": 8192,
        "NUM_POSTERIOR_SAMPLES": 2000,
        "NPE_DATA_DEVICE": "cpu",
        "PAULI_OBS_MODE": "random",
        "SAVE_NPE_TRAINING_NPZ": True,
        "MAX_N_QUBITS_DENSE_RHO_PLOTS": N_QUBITS,
    })
elif _PRESET != "random_state":
    raise ValueError(f"Unknown QST_PRESET={_PRESET!r}; use 'random_state' or 'prep_sweep_cholesky'")


def _brickwall_layout_from_cfg(cfg: dict) -> tuple[bool, bool]:
    """``(per_qubit_prep, unified_ry_rx)`` of the circuit-angle latents."""
    per_qubit = bool(cfg.get("BRICKWALL_PER_QUBIT_PREP", False))
    unified = bool(cfg.get("BRICKWALL_UNIFIED_RY_RX", False))
    if unified and not per_qubit:
        raise ValueError("BRICKWALL_UNIFIED_RY_RX requires BRICKWALL_PER_QUBIT_PREP=True")
    return per_qubit, unified


def _ansatz_layers_from_cfg(cfg: dict, *, for_eval: bool = False) -> int:
    """Brickwall layer count for training, or for the held-out random circuits."""
    if for_eval and cfg.get("RANDOM_EVAL_ANSATZ_LAYERS") is not None:
        return max(1, int(cfg["RANDOM_EVAL_ANSATZ_LAYERS"]))
    return max(1, int(cfg.get("BRICKWALL_ANSATZ_LAYERS", 1)))


def _cnot_topology_from_cfg(cfg: dict) -> str:
    return str(cfg.get("BRICKWALL_CNOT_TOPOLOGY", "linear")).lower()


def _num_params_from_config(cfg: dict) -> int:
    """Latent dimension implied by ``STATE_PARAM`` and the brickwall layout."""
    state_param = str(cfg["STATE_PARAM"]).lower()
    n_qubits = int(cfg["N_QUBITS"])
    if state_param == "cholesky":
        return 2 * 2**n_qubits
    if state_param == "circuit_angles":
        per_qubit, unified = _brickwall_layout_from_cfg(cfg)
        return num_params_brickwall_angles(
            n_qubits,
            per_qubit_prep=per_qubit,
            unified_ry_rx=unified,
            ansatz_layers=_ansatz_layers_from_cfg(cfg),
        )
    raise ValueError(f"Unknown STATE_PARAM: {state_param!r}; use 'circuit_angles' or 'cholesky'")


NUM_PARAMS = _num_params_from_config(CONFIG)


def _print_normalizing_flow_param_count(n_obs: int) -> None:
    """Instantiate the flow on CPU and report its parameter count (before the long simulations)."""
    if CONFIG.get("STANDARDIZE_X", True):
        embed = _StandardizeX(
            torch.zeros(n_obs, dtype=torch.float32),
            torch.ones(n_obs, dtype=torch.float32),
        )
    else:
        embed = _PassThroughX()
    density_estimator = _build_density_estimator(CONFIG, n_obs, embed)
    flow = density_estimator(
        torch.zeros(2, NUM_PARAMS, dtype=torch.float32),
        torch.zeros(2, n_obs, dtype=torch.float32),
    )
    n_flow = sum(p.numel() for p in flow.parameters())
    print(
        f"Normalizing flow ({CONFIG['FLOW_MODEL']}): hidden={CONFIG['FLOW_HIDDEN_FEATURES']}, "
        f"transforms={CONFIG['NUM_TRANSFORMS']}, θ_dim={NUM_PARAMS}, x_dim={n_obs} → {n_flow:,} parameters"
    )


def _config_value_jsonable(obj: object) -> object:
    """Recursively convert CONFIG values for JSON / torch.save."""
    if isinstance(obj, dict):
        return {k: _config_value_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_config_value_jsonable(v) for v in obj]
    if obj is None or isinstance(obj, (bool, str, int, float)):
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _save_config_checkpoint(results_dir: str, cfg: dict) -> None:
    """Write ``config_checkpoint.json`` and ``config_checkpoint.pt`` under ``results_dir``."""
    checkpoint = {
        "config": _config_value_jsonable(dict(cfg)),
        "meta": {
            "saved_at": datetime.now().isoformat(),
            "num_params": int(NUM_PARAMS),
            "dim_hilbert": int(DIM),
        },
    }
    base = os.path.join(results_dir, "config_checkpoint")
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)
    torch.save(checkpoint, base + ".pt")
    print(f"  Saved config checkpoint → {base}.json / {base}.pt")


def _pick_pauli_strings() -> tuple[list[str], str, int]:
    obs_mode = str(CONFIG["PAULI_OBS_MODE"]).lower()
    if obs_mode == "random":
        n_rand = int(CONFIG.get("N_RANDOM_PAULIS", 100))
        max_paulis = 4**N_QUBITS - 1
        if n_rand > max_paulis:
            print(f"  N_RANDOM_PAULIS capped: {n_rand} -> {max_paulis} (distinct non-identity Paulis)")
            n_rand = max_paulis
        pseed = int(CONFIG.get("PAULI_DRAW_SEED", 51042))
        pauli_strings = random_pauli_strings(N_QUBITS, n_rand, np.random.default_rng(pseed))
        obs_desc = f"random ({len(pauli_strings)} strings, seed={pseed})"
    elif obs_mode == "weight1_nn":
        pauli_strings = pauli_strings_weight1_nn(N_QUBITS)
        obs_desc = "weight-1 + NN weight-2"
    else:
        raise ValueError(f"Unknown PAULI_OBS_MODE={obs_mode!r}; use 'random' or 'weight1_nn'")
    if len(pauli_strings) != len(set(pauli_strings)):
        raise AssertionError("Pauli observation list contains duplicates")
    return pauli_strings, obs_desc, len(pauli_strings)


def _simulate_obs(
    theta_bank: np.ndarray,
    pauli_strings: list[str],
    cfg: dict,
    *,
    shots: int | None,
    seed: int,
    desc: str,
    ansatz_layers: int | None = None,
    unified_ry_rx: bool | None = None,
) -> np.ndarray:
    """Pauli expectations ⟨P⟩ of the states encoded by the rows of ``theta_bank`` (the NPE forward model)."""
    theta_np = np.asarray(theta_bank, dtype=np.float32)
    if theta_np.ndim == 1:
        theta_np = theta_np.reshape(1, -1)
    if unified_ry_rx is None:
        unified_ry_rx = _brickwall_layout_from_cfg(cfg)[1]
    return simulate_parallel(
        theta_np,
        pauli_strings,
        shots=shots,
        seed=int(seed),
        desc=desc,
        state_param=str(cfg["STATE_PARAM"]).lower(),
        n_qubits=int(cfg["N_QUBITS"]),
        tensor_bond_dim=int(cfg.get("TENSOR_BOND_DIM", 2)),
        tn_avoid_dense=bool(cfg.get("TN_AVOID_DENSE", True)),
        brickwall_ansatz_layers=_ansatz_layers_from_cfg(cfg) if ansatz_layers is None else int(ansatz_layers),
        brickwall_unified_ry_rx=bool(unified_ry_rx),
        brickwall_cnot_topology=_cnot_topology_from_cfg(cfg),
    ).astype(np.float32)


def _add_shot_noise(x: np.ndarray, shots: int | None, seeds) -> np.ndarray:
    """Replace each exact row of ``x`` by a binomial shot estimate (row ``i`` uses RNG seed ``seeds[i]``)."""
    if shots is None:
        return x
    for i in range(len(x)):
        rng = np.random.default_rng(int(seeds[i]))
        p_plus = np.clip(0.5 * (1.0 + x[i]), 0.0, 1.0)
        freq = rng.binomial(int(shots), p_plus) / float(shots)
        x[i] = (2.0 * freq - 1.0).astype(np.float32)
    return x


def _truth_state(ty: float, tx: float, cfg: dict, *, dense: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
    """``(θ, dense ψ)`` of the brickwall state with prep angles ``(ty, tx)``.

    ψ is always returned for ``cholesky``; for ``circuit_angles`` only when ``dense`` and n is within the
    dense-plot limit (fidelities there use θ, not ψ).
    """
    per_qubit, unified = _brickwall_layout_from_cfg(cfg)
    layers = _ansatz_layers_from_cfg(cfg)
    n = int(cfg["N_QUBITS"])
    state_param = str(cfg["STATE_PARAM"]).lower()
    theta = prep_to_theta(
        ty, tx, n_qubits=n, state_param=state_param,
        per_qubit_prep=per_qubit, unified_ry_rx=unified, ansatz_layers=layers,
    )
    if state_param == "cholesky":
        return theta, brickwall_rx_ry_state_vector(n, theta_y=ty, theta_x=tx)
    if not dense or n > int(cfg.get("MAX_N_QUBITS_DENSE_RHO_PLOTS", n)):
        return theta, None
    psi = theta_to_psi_brickwall_angles(
        theta, n_qubits=n, ansatz_layers=layers, unified_ry_rx=unified, cnot_topology=_cnot_topology_from_cfg(cfg),
    )
    return theta, psi


def _fidelity_kwargs(cfg: dict) -> dict:
    """Decoding options shared by ``viz.posterior_sample_fidelities`` and ``viz.plot_all``."""
    return dict(
        state_param=str(cfg["STATE_PARAM"]).lower(),
        tensor_bond_dim=int(cfg.get("TENSOR_BOND_DIM", 2)),
        ansatz_layers=_ansatz_layers_from_cfg(cfg),
        unified_ry_rx=_brickwall_layout_from_cfg(cfg)[1],
        cnot_topology=_cnot_topology_from_cfg(cfg),
    )


def _write_run_files(
    results_dir: str, pauli_strings: list[str], n_obs: int, cfg: dict, *, snapshot: bool = True,
) -> None:
    os.makedirs(results_dir, exist_ok=True)
    _save_config_checkpoint(results_dir, cfg)
    with open(os.path.join(results_dir, "pauli_strings.txt"), "w") as f:
        f.write("\n".join(pauli_strings))
    if not snapshot:
        return
    with open(os.path.join(results_dir, "config_snapshot.txt"), "w") as f:
        f.write(f"run at {datetime.now().isoformat()}\n")
        for k, v in cfg.items():
            f.write(f"{k}: {v}\n")
        f.write(f"num_params: {NUM_PARAMS}\n")
        f.write(f"n_obs: {n_obs}\n")
        f.write(f"pauli_count: {len(pauli_strings)}\n")


def _run_one_qst(
    results_dir: str,
    pauli_strings: list[str],
    n_obs: int,
    cfg: dict,
) -> dict[str, float]:
    """Train one NPE on the reference state's x_obs, plot its posterior, then evaluate on random circuits."""
    _write_run_files(results_dir, pauli_strings, n_obs, cfg)

    ty = float(cfg.get("PREP_THETA_Y", np.pi / 4.0))
    tx = float(cfg.get("PREP_THETA_X", np.pi / 4.0))
    theta_truth, psi_true = _truth_state(ty, tx, cfg)
    print(f"Truth state: brickwall Rx Ry  (θ_y={ty:.5g} rad, θ_x={tx:.5g} rad)")

    print(f"Building conditioning x_obs with simulate_parallel (shots={cfg['SHOTS']})...")
    x_obs_np = _simulate_obs(
        theta_truth, pauli_strings, cfg, shots=cfg["SHOTS"], seed=int(cfg["OBS_SEED"]), desc="obs",
    )[0]
    np.savetxt(os.path.join(results_dir, "x_obs.csv"), x_obs_np.reshape(1, -1), delimiter=",")

    x_obs = torch.as_tensor(x_obs_np, dtype=torch.float32, device=cfg["DEVICE"])
    posterior, samples, _est = train_npe(
        cfg,
        x_obs=x_obs,
        num_params=NUM_PARAMS,
        n_obs=n_obs,
        pauli_strings=pauli_strings,
        results_dir=results_dir,
    )
    samp_path = os.path.join(results_dir, "posterior_samples.csv")
    np.savetxt(samp_path, samples, delimiter=",")
    print(f"Saved posterior samples → {samp_path}")

    fstats = viz.plot_all(
        results_dir=results_dir,
        posterior_samples=samples,
        psi_true=psi_true,
        max_n_qubits_dense_rho_plots=int(cfg.get("MAX_N_QUBITS_DENSE_RHO_PLOTS", cfg["N_QUBITS"])),
        prep_theta=theta_truth,
        n_qubits=int(cfg["N_QUBITS"]),
        **_fidelity_kwargs(cfg),
    )
    _run_random_parameter_eval(
        posterior=posterior,
        output_dir=results_dir,
        pauli_strings=pauli_strings,
        cfg=cfg,
        device=str(cfg["DEVICE"]),
    )
    return fstats


def _run_random_parameter_eval(
    *,
    posterior,
    output_dir: str,
    pauli_strings: list[str],
    cfg: dict,
    device: str,
) -> None:
    """Evaluate the posterior on held-out random circuits (circuit_angles only).

    The training prior width is set by ``BRICKWALL_PER_QUBIT_PREP``; the held-out states use
    ``RANDOM_EVAL_PER_QUBIT_PREP`` / ``RANDOM_EVAL_UNIFIED_RY_RX`` and are not restricted to that prior.
    Writes ``random_parameter_eval.npz`` and ``qst_random_state_fidelity_hist.pdf``.
    """
    n_eval = int(cfg.get("RANDOM_EVAL_N_STATES", 0))
    if n_eval <= 0:
        return
    state_param = str(cfg["STATE_PARAM"]).lower()
    if state_param != "circuit_angles":
        print("  Skipping random-parameter eval (requires STATE_PARAM='circuit_angles').")
        return

    n_post = int(cfg["NUM_POSTERIOR_SAMPLES"])
    n_q = int(cfg["N_QUBITS"])
    y_range = tuple(float(v) for v in cfg["SWEEP_THETA_Y"])
    x_range = tuple(float(v) for v in cfg["SWEEP_THETA_X"])
    eval_layers = _ansatz_layers_from_cfg(cfg, for_eval=True)
    train_layers = _ansatz_layers_from_cfg(cfg)
    train_per_qubit, train_unified = _brickwall_layout_from_cfg(cfg)
    random_per_qubit = bool(
        cfg.get("RANDOM_EVAL_PER_QUBIT_PREP", cfg.get("BRICKWALL_PER_QUBIT_PREP", False))
    )
    random_unified = bool(
        cfg.get("RANDOM_EVAL_UNIFIED_RY_RX", cfg.get("BRICKWALL_UNIFIED_RY_RX", False))
    )
    if random_unified and not random_per_qubit:
        raise ValueError("RANDOM_EVAL_UNIFIED_RY_RX requires RANDOM_EVAL_PER_QUBIT_PREP=True")
    eval_seed = int(cfg.get("SIM_SEED", 42)) + 10_000_019
    obs_seed0 = int(cfg.get("OBS_SEED", 123)) + 20_000_003
    dedup_decimals = int(cfg.get("THETA_DEDUP_DECIMALS", 12))
    rng = np.random.default_rng(eval_seed)

    eval_width = num_params_brickwall_angles(
        n_q, per_qubit_prep=random_per_qubit, unified_ry_rx=random_unified, ansatz_layers=eval_layers
    )
    forbidden_keys: set[bytes] | None = None
    train_theta_path = os.path.join(output_dir, "training_theta.npy")
    if os.path.isfile(train_theta_path):
        train_theta = np.load(train_theta_path, mmap_mode="r")
        if train_theta.ndim == 2 and train_theta.shape[1] == eval_width:
            forbidden_keys = brickwall_theta_keys(train_theta, decimals=dedup_decimals)
            print(
                f"  Random eval dedup: excluding {len(forbidden_keys)} training θ keys "
                f"(decimals={dedup_decimals})"
            )
        else:
            print(
                f"  Random eval dedup: skipped (training θ shape {train_theta.shape} ≠ eval width {eval_width})"
            )
    else:
        print(f"  Random eval dedup: no {train_theta_path}; eval θ unique within draw only")
    print(
        f"  Random eval prep: L={eval_layers} per_qubit={random_per_qubit} unified_ry_rx={random_unified} "
        f"(training L={train_layers} per_qubit={train_per_qubit} unified={train_unified}, "
        f"NPE latent dim={NUM_PARAMS})"
    )
    theta_bank, n_rejected = sample_brickwall_angle_theta_unique(
        n_q,
        n_eval,
        rng,
        theta_y_range=y_range,
        theta_x_range=x_range,
        per_qubit_prep=random_per_qubit,
        unified_ry_rx=random_unified,
        ansatz_layers=eval_layers,
        forbidden_keys=forbidden_keys,
        dedup_decimals=dedup_decimals,
    )
    if n_rejected:
        print(f"  Random eval dedup: rejected {n_rejected} duplicate θ draws")

    print(
        f"\n[RANDOM EVAL] {n_eval} held-out {eval_layers}-layer random circuit states "
        f"({theta_bank.shape[1]} circuit params each), {n_post} posterior samples/state"
    )
    t_obs = time.perf_counter()
    x_eval = _simulate_obs(
        theta_bank, pauli_strings, cfg, shots=None, seed=obs_seed0, desc="eval obs",
        ansatz_layers=eval_layers, unified_ry_rx=random_unified,
    )
    x_eval = _add_shot_noise(x_eval, cfg.get("SHOTS"), obs_seed0 + np.arange(n_eval))
    print(f"  Observables: {time.perf_counter() - t_obs:.1f}s ({n_eval} states)")

    fid_kw = dict(_fidelity_kwargs(cfg), ansatz_layers=eval_layers, unified_ry_rx=random_unified)
    all_f = np.empty(n_eval * n_post, dtype=np.float64)
    t_post = time.perf_counter()
    for i in range(n_eval):
        x_t = torch.as_tensor(x_eval[i], dtype=torch.float32, device=device).reshape(1, -1)
        samples = draw_posterior_samples(posterior, x_t, n_post)
        fids = viz.posterior_sample_fidelities(
            samples, None, n_qubits=n_q, prep_theta=theta_bank[i], **fid_kw,
        )
        all_f[i * n_post:(i + 1) * n_post] = np.asarray(fids, dtype=np.float64).ravel()[:n_post]
        if (i + 1) % max(1, n_eval // 10) == 0 or i + 1 == n_eval:
            print(f"  {eval_layers}-layer random circuit {i + 1:03d}/{n_eval}")
    print(f"  Posterior + fidelity: {time.perf_counter() - t_post:.1f}s ({n_eval}×{n_post} samples)")

    np.savez_compressed(
        os.path.join(output_dir, "random_parameter_eval.npz"),
        theta=theta_bank.astype(np.float32),
        x_obs_raw=x_eval,
        fidelity=all_f,
        eval_seed=np.int64(eval_seed),
        obs_seed0=np.int64(obs_seed0),
        num_posterior_samples=np.int32(n_post),
        per_qubit_prep=np.bool_(random_per_qubit),
        unified_ry_rx=np.bool_(random_unified),
        train_prior_per_qubit=np.bool_(train_per_qubit),
        ansatz_layers=np.int32(eval_layers),
        theta_dedup_rejected=np.int64(n_rejected),
    )
    viz.plot_random_state_fidelity_histogram(output_dir, all_f)


def _main_sweep(pauli_strings: list[str], n_obs: int, obs_desc: str) -> None:
    """Train one NPE on the reference state, then test it on a grid of prep angles (θ_y, θ_x)."""
    ny = int(CONFIG["SWEEP_N_Y"])
    nx = int(CONFIG["SWEEP_N_X"])
    ly, hy = CONFIG["SWEEP_THETA_Y"]
    lx, hx = CONFIG["SWEEP_THETA_X"]
    ty_vals = np.linspace(float(ly), float(hy), ny, endpoint=False, dtype=np.float64)
    tx_vals = np.linspace(float(lx), float(hx), nx, endpoint=False, dtype=np.float64)
    device = str(CONFIG["DEVICE"])
    n_post = int(CONFIG["NUM_POSTERIOR_SAMPLES"])
    base_obs_seed = int(CONFIG["OBS_SEED"])
    shots = CONFIG.get("SHOTS")
    save_x_grid = bool(CONFIG.get("SAVE_SWEEP_X_OBS_GRID", True))

    sweep_dir = os.path.join("sbi_qst_results", datetime.now().strftime("qst_prep_sweep_%Y-%m-%d_%H-%M-%S"))
    _write_run_files(sweep_dir, pauli_strings, n_obs, CONFIG, snapshot=False)
    with open(os.path.join(sweep_dir, "sweep_readme.txt"), "w") as f:
        f.write("One NPE trained on reference PREP_THETA_Y/X; then the same posterior conditioned on\n")
        f.write("simulated x_obs for each grid (theta_y, theta_x). Standardisation uses training stats.\n")
        f.write(f"Pauli design: {obs_desc}\n")
        f.write(f"Grid ny={ny} nx={nx}; median infidelity = 1 - median(F_i); NUM_POSTERIOR_SAMPLES={n_post} each.\n")
        f.write(f"sweep_fidelity_grid.npz: fidelity shape (ny,nx,{n_post}) — all grid posterior fidelities.\n")

    ty0 = float(CONFIG["PREP_THETA_Y"])
    tx0 = float(CONFIG["PREP_THETA_X"])
    print(f"\n[SWEEP] Train single NPE with reference prep θ_y={ty0:.5g} rad, θ_x={tx0:.5g} rad")
    theta_ref, psi_ref = _truth_state(ty0, tx0, CONFIG)
    print(f"[SWEEP] Reference conditioning x_obs (shots={shots})")
    x_ref_np = _simulate_obs(
        theta_ref, pauli_strings, CONFIG, shots=shots, seed=base_obs_seed, desc="obs",
    )[0]
    np.savetxt(os.path.join(sweep_dir, "x_obs_reference.csv"), x_ref_np.reshape(1, -1), delimiter=",")
    x_ref = torch.as_tensor(x_ref_np, dtype=torch.float32, device=device)

    posterior, samples_ref, _est = train_npe(
        CONFIG,
        x_obs=x_ref,
        num_params=NUM_PARAMS,
        n_obs=n_obs,
        pauli_strings=pauli_strings,
        results_dir=sweep_dir,
    )
    np.savetxt(os.path.join(sweep_dir, "posterior_samples_reference.csv"), samples_ref, delimiter=",")
    fstats_ref = viz.plot_all(
        results_dir=sweep_dir,
        posterior_samples=samples_ref,
        psi_true=psi_ref,
        max_n_qubits_dense_rho_plots=int(CONFIG.get("MAX_N_QUBITS_DENSE_RHO_PLOTS", N_QUBITS)),
        prep_theta=theta_ref,
        n_qubits=N_QUBITS,
        **_fidelity_kwargs(CONFIG),
    )
    print(
        f"[SWEEP] Reference posterior: median(F_i)={fstats_ref['fidelity_median']:.6f}  "
        f"(see figures in {sweep_dir}/)"
    )

    grid_seeds = np.array(
        [base_obs_seed + 1 + iy * 100_003 + ix * 7919 for iy in range(ny) for ix in range(nx)],
        dtype=np.int64,
    )
    states = [_truth_state(float(ty_vals[iy]), float(tx_vals[ix]), CONFIG, dense=False) for iy in range(ny) for ix in range(nx)]
    theta_grid = np.stack([theta for theta, _psi in states])

    print(f"[SWEEP] Batched eval for {ny}×{nx}={ny * nx} grid states...")
    t_obs = time.perf_counter()
    x_flat = _simulate_obs(
        theta_grid, pauli_strings, CONFIG, shots=None, seed=base_obs_seed + 1, desc="grid obs",
    )
    x_flat = _add_shot_noise(x_flat, shots, grid_seeds)
    print(f"[SWEEP] batch done in {time.perf_counter() - t_obs:.1f}s")

    fid_kw = _fidelity_kwargs(CONFIG)
    med_inf = np.empty((ny, nx), dtype=np.float64)
    fidelity_grid = np.empty((ny, nx, n_post), dtype=np.float64)
    for iy in range(ny):
        for ix in range(nx):
            idx = iy * nx + ix
            theta_cell, psi_cell = states[idx]
            x_t = torch.as_tensor(x_flat[idx].astype(np.float32), dtype=torch.float32, device=device).reshape(1, -1)
            samples = draw_posterior_samples(posterior, x_t, n_post)
            fids = viz.posterior_sample_fidelities(
                samples, psi_cell, n_qubits=N_QUBITS, prep_theta=theta_cell, **fid_kw,
            )
            fids = np.asarray(fids, dtype=np.float64).ravel()[:n_post]
            fidelity_grid[iy, ix, : fids.size] = fids
            if fids.size < n_post:
                fidelity_grid[iy, ix, fids.size:] = np.nan
            med_f = float(np.median(fids)) if len(fids) else 0.0
            med_inf[iy, ix] = 1.0 - med_f
            print(
                f"  grid iy={iy} ix={ix}  θ_y={ty_vals[iy]:.6f} rad  θ_x={tx_vals[ix]:.6f} rad  "
                f"median(F_i)={med_f:.6f}  infidelity={med_inf[iy, ix]:.6f}"
            )

    if save_x_grid:
        np.savez_compressed(
            os.path.join(sweep_dir, "sweep_x_obs_grid.npz"),
            theta_y=ty_vals,
            theta_x=tx_vals,
            x_obs_raw=x_flat.reshape(ny, nx, n_obs).astype(np.float32),
            seed=grid_seeds.reshape(ny, nx),
        )
    np.savez_compressed(
        os.path.join(sweep_dir, "sweep_median_infidelity.npz"),
        theta_y=ty_vals,
        theta_x=tx_vals,
        median_infidelity=med_inf,
    )
    fidelity_flat = fidelity_grid.reshape(-1)
    fidelity_flat = fidelity_flat[np.isfinite(fidelity_flat)]
    f_mean = float(np.mean(fidelity_flat)) if fidelity_flat.size else 0.0
    f_median = float(np.median(fidelity_flat)) if fidelity_flat.size else 0.0
    f_max = float(np.max(fidelity_flat)) if fidelity_flat.size else 0.0
    np.savez_compressed(
        os.path.join(sweep_dir, "sweep_fidelity_grid.npz"),
        theta_y=ty_vals,
        theta_x=tx_vals,
        fidelity=fidelity_grid,
        num_posterior_samples=np.int32(n_post),
        fidelity_mean=np.float64(f_mean),
        fidelity_median=np.float64(f_median),
        fidelity_max=np.float64(f_max),
    )
    print(
        f"[SWEEP] Saved sweep_fidelity_grid.npz — {ny}×{nx}×{n_post} = {ny * nx * n_post} "
        f"fidelities; pooled F_mean={f_mean:.6f}  F_median={f_median:.6f}  F_max={f_max:.6f}"
    )
    viz.plot_prep_angle_sweep_median_infidelity(sweep_dir, ty_vals, tx_vals, med_inf)
    viz.plot_random_state_fidelity_histogram(sweep_dir, fidelity_grid, filename="qst_sweep_fidelity_hist.pdf")
    _run_random_parameter_eval(
        posterior=posterior,
        output_dir=sweep_dir,
        pauli_strings=pauli_strings,
        cfg=CONFIG,
        device=device,
    )
    print(f"\nSweep done. Root: {sweep_dir}/")


def _print_device_diagnostics() -> None:
    """Which pipeline stages use the GPU."""
    cuda = torch.cuda.is_available()
    print("\n=== Runtime / device ===")
    print(f"  CONFIG['DEVICE']     = {CONFIG['DEVICE']}  (NPE model + train)")
    print(f"  NPE_DATA_DEVICE      = {CONFIG.get('NPE_DATA_DEVICE', 'cpu')}  (θ,x store for sbi)")
    print(f"  torch.cuda.is_available() = {cuda}")
    if cuda:
        print(f"  CUDA device          = {torch.cuda.get_device_name(0)}")
    else:
        print("  WARNING: no CUDA → simulate_parallel and NPE both fall back to CPU.")
    print(f"  PRIOR_TYPE           = {CONFIG['PRIOR_TYPE']}")
    print(f"  STATE_PARAM          = {CONFIG['STATE_PARAM']}")
    print(f"  TN_AVOID_DENSE       = {CONFIG['TN_AVOID_DENSE']} (n={N_QUBITS}, 2^n={DIM})")
    print(f"  NPE_N_SIMS           = {CONFIG['NPE_N_SIMS']}")
    print("========================\n")


def _main():
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    _print_device_diagnostics()

    pauli_strings, obs_desc, n_obs = _pick_pauli_strings()
    print(f"Pauli observables: {n_obs} ({obs_desc})")
    if CONFIG["STATE_PARAM"] == "circuit_angles":
        if CONFIG["BRICKWALL_UNIFIED_RY_RX"]:
            prep_desc = "per-qubit unified θ (Ry/Rx share θ_q)"
        elif CONFIG["BRICKWALL_PER_QUBIT_PREP"]:
            prep_desc = "per-qubit θ_y, θ_x"
        else:
            prep_desc = "shared θ_y, θ_x"
        print(f"Circuit-ansatz latent: brickwall Rx/Ry + CNOT, {prep_desc}, num_params={NUM_PARAMS}")
    _print_normalizing_flow_param_count(n_obs)

    if CONFIG["PREP_ANGLE_SWEEP"]:
        _main_sweep(pauli_strings, n_obs, obs_desc)
        return

    results_dir = os.path.join("sbi_qst_results", datetime.now().strftime("qst_%Y-%m-%d_%H-%M-%S"))
    fstats = _run_one_qst(results_dir, pauli_strings, n_obs, CONFIG)
    print("Inferred fidelities (eval / sanity check — uses true |ψ>):")
    print(f"  mean(F_i) = average over posterior samples = {fstats['fidelity_mean_samples']:.6f}")
    print(f"  median(F_i)= {fstats['fidelity_median']:.6f}")
    print(f"  max_i F_i  = {fstats['fidelity_best']:.6f}")
    print(f"\nDone. Results in {results_dir}/")


if __name__ == "__main__":
    if multiprocessing.current_process().name == "MainProcess":
        _main()
