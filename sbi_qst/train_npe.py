"""Single-round amortised NPE (normalising flow): training-set simulation, training and posterior sampling."""

from __future__ import annotations

import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as td
from sbi.inference import NPE
from sbi.inference.posteriors import DirectPosterior
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform

from sbi_qst.physics import sample_brickwall_angle_theta_unique
from sbi_qst.simulate import simulate_parallel


class PermissivePrior:
    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        return torch.zeros(theta.shape[0], device=theta.device)


class _StandardizeX(nn.Module):
    """Per-dimension (x - mean) / std for the conditioning vector (train stats from simulations)."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean.float().reshape(1, -1).clone())
        s = std.float().reshape(1, -1).clone()
        self.register_buffer("std", torch.clamp(s, min=1e-6))
        # sbi checks embedding_net has ≥1 Parameter (buffers alone are not enough).
        self.register_parameter("_device_anchor", nn.Parameter(torch.zeros(1), requires_grad=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return (x - self.mean) / self.std


class _PassThroughX(nn.Module):
    """Identity with a dummy param so sbi accepts this as embedding_net."""

    def __init__(self):
        super().__init__()
        self.register_parameter("_device_anchor", nn.Parameter(torch.zeros(1), requires_grad=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _build_density_estimator(
    config: dict,
    n_obs: int,
    embedding_net: nn.Module,
):
    return posterior_nn(
        model="zuko_nsf",
        hidden_features=config["FLOW_HIDDEN_FEATURES"],
        num_transforms=config["NUM_TRANSFORMS"],
        embedding_net=embedding_net,
        num_bins=config.get("NSF_NUM_BINS", 8),
    )


def _build_posterior(inference, estimator, prior, device, x_obs_squeezed):
    posterior = DirectPosterior(posterior_estimator=estimator, prior=prior, device=device)
    return posterior.set_default_x(x_obs_squeezed)


def _make_prior(config: dict, num_params: int):
    """Prior of θ (shape [*, num_params]): standard Gaussian (cholesky) or uniform brickwall angles."""
    prior_type = str(config["PRIOR_TYPE"]).lower()
    device = config["DEVICE"]

    if prior_type == "gaussian":
        loc = torch.zeros(num_params, device=device, dtype=torch.float32)
        scale = torch.ones(num_params, device=device, dtype=torch.float32)
        return td.Independent(td.Normal(loc=loc, scale=scale), 1)

    if prior_type == "brickwall_angles":
        n_q = int(config["N_QUBITS"])
        per_qubit = bool(config.get("BRICKWALL_PER_QUBIT_PREP", False))
        unified = bool(config.get("BRICKWALL_UNIFIED_RY_RX", False))
        y_lo, y_hi = (float(v) for v in config["SWEEP_THETA_Y"])
        x_lo, x_hi = (float(v) for v in config["SWEEP_THETA_X"])
        if unified and per_qubit:
            low = torch.tensor([y_lo] * n_q, device=device, dtype=torch.float32)
            high = torch.tensor([y_hi] * n_q, device=device, dtype=torch.float32)
        elif per_qubit:
            low = torch.tensor([y_lo] * n_q + [x_lo] * n_q, device=device, dtype=torch.float32)
            high = torch.tensor([y_hi] * n_q + [x_hi] * n_q, device=device, dtype=torch.float32)
        else:
            low = torch.tensor([y_lo, x_lo], device=device, dtype=torch.float32)
            high = torch.tensor([y_hi, x_hi], device=device, dtype=torch.float32)
        if low.numel() == num_params:
            return BoxUniform(low=low, high=high)
        # Multi-layer ansatz: θ is drawn by the custom sampler; this box is only sbi's bookkeeping prior.
        return BoxUniform(
            low=torch.full((num_params,), float(config.get("PRIOR_LOW", -3.0)), device=device),
            high=torch.full((num_params,), float(config.get("PRIOR_HIGH", 3.0)), device=device),
        )

    raise ValueError(f"Unknown PRIOR_TYPE={prior_type!r}; use 'gaussian' or 'brickwall_angles'.")


def _sample_training_theta(config: dict, num_params: int, n_sims: int, prior) -> torch.Tensor:
    """Training θ: unique brickwall angles (circuit_angles) or ``prior.sample`` (cholesky)."""
    device = config["DEVICE"]
    state_param = str(config["STATE_PARAM"]).lower()

    if state_param == "circuit_angles":
        if str(config["PRIOR_TYPE"]).lower() != "brickwall_angles":
            raise ValueError("STATE_PARAM='circuit_angles' requires PRIOR_TYPE='brickwall_angles'.")
        rng = np.random.default_rng(int(config.get("SIM_SEED", 42)))
        per_qubit_prep = bool(config.get("BRICKWALL_PER_QUBIT_PREP", False))
        unified_ry_rx = bool(config.get("BRICKWALL_UNIFIED_RY_RX", False))
        ansatz_layers = max(1, int(config.get("BRICKWALL_ANSATZ_LAYERS", 1)))
        dedup_decimals = int(config.get("THETA_DEDUP_DECIMALS", 12))
        theta_np, n_rejected = sample_brickwall_angle_theta_unique(
            int(config["N_QUBITS"]),
            n_sims,
            rng,
            theta_y_range=tuple(float(v) for v in config["SWEEP_THETA_Y"]),
            theta_x_range=tuple(float(v) for v in config["SWEEP_THETA_X"]),
            per_qubit_prep=per_qubit_prep,
            unified_ry_rx=unified_ry_rx,
            ansatz_layers=ansatz_layers,
            dedup_decimals=dedup_decimals,
        )
        if theta_np.shape[1] != num_params:
            raise ValueError(f"brickwall angle theta width {theta_np.shape[1]} != num_params {num_params}")
        if unified_ry_rx:
            prep_desc = f"per-qubit unified θ (Ry/Rx share θ_q), L={ansatz_layers}"
        elif per_qubit_prep:
            prep_desc = f"per-qubit θ_y, θ_x, L={ansatz_layers}"
        else:
            prep_desc = f"shared θ_y, θ_x, L={ansatz_layers}"
        print(
            f"  Training θ: brickwall circuit angles ({prep_desc}) "
            f"(num_params={theta_np.shape[1]}, dedup decimals={dedup_decimals}, rejected={n_rejected})"
        )
        return torch.as_tensor(theta_np, dtype=torch.float32, device=device)

    return prior.sample((n_sims,)).to(dtype=torch.float32)


def draw_posterior_samples(posterior, x: torch.Tensor, n_post: int) -> np.ndarray:
    """``n_post`` posterior samples conditioned on ``x``, drawn in batches (float32, shape (n_post, θ_dim))."""
    if x.dim() == 1:
        x = x.unsqueeze(0)
    batches: list[torch.Tensor] = []
    need = n_post
    bs = min(512, max(128, n_post // 5))
    while need > 0:
        b = posterior.sample((min(bs, need),), x=x, show_progress_bars=False)
        batches.append(b.detach().cpu())
        need -= b.shape[0]
        if len(batches) * bs > n_post * 20:
            warnings.warn(
                f"draw_posterior_samples: safety break triggered — "
                f"collected {sum(t.shape[0] for t in batches)} of {n_post} requested samples.",
                RuntimeWarning, stacklevel=2,
            )
            break
    samples = torch.cat(batches, dim=0)[:n_post]
    if samples.shape[0] < n_post:
        warnings.warn(
            f"draw_posterior_samples: returning {samples.shape[0]} samples instead of {n_post}.",
            RuntimeWarning, stacklevel=2,
        )
    return samples.numpy().astype(np.float32)


def train_npe(
    config: dict,
    x_obs: torch.Tensor,
    num_params: int,
    n_obs: int,
    pauli_strings: list[str],
    results_dir: str,
) -> tuple[object, np.ndarray, object | None]:
    device = config["DEVICE"]
    n_sims = int(config["NPE_N_SIMS"])
    standardize_x = bool(config.get("STANDARDIZE_X", True))

    state_param = str(config["STATE_PARAM"]).lower()
    tensor_bond_dim = int(config.get("TENSOR_BOND_DIM", 2))
    n_qubits = int(config["N_QUBITS"])
    tn_avoid_dense = bool(config.get("TN_AVOID_DENSE", False))

    prior = _make_prior(config, num_params)

    training_batch_size = min(int(config["BATCH_SIZE"]), n_sims)
    npe_data_device = str(config.get("NPE_DATA_DEVICE", "cpu"))
    if npe_data_device in ("cuda", "gpu"):
        npe_data_device = "cuda" if torch.cuda.is_available() else "cpu"
    train_on_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if not train_on_cuda:
        print("  WARNING: DEVICE is not CUDA — NPE will train on CPU.")

    train_kwargs = dict(
        training_batch_size=training_batch_size,
        learning_rate=config["LR"],
        max_num_epochs=config["EPOCHS"],
        stop_after_epochs=config["STOP_AFTER_EPOCHS"],
        show_train_summary=True,
        clip_max_norm=config["CLIP_MAX_NORM"],
        # Data on CPU + pin_memory speeds H2D when model trains on CUDA (sbi GPU how-to).
        dataloader_kwargs={
            "pin_memory": train_on_cuda and npe_data_device == "cpu",
            "num_workers": 0,
        },
    )

    print(f"\nDrawing {n_sims} prior samples and simulating Pauli shot frequencies...")
    t_theta = time.perf_counter()
    theta_gpu = _sample_training_theta(config, num_params, n_sims, prior)
    print(f"  θ on {theta_gpu.device} ({time.perf_counter() - t_theta:.1f}s total incl. H2D)")
    if state_param == "circuit_angles":
        train_theta_path = os.path.join(results_dir, "training_theta.npy")
        np.save(train_theta_path, theta_gpu.detach().cpu().numpy().astype(np.float32, copy=False))
        print(f"  Saved training θ for held-out dedup → {train_theta_path}")
    t_sim = time.perf_counter()
    x_np = simulate_parallel(
        theta_gpu,
        pauli_strings,
        shots=config["SHOTS"],
        seed=int(config["SIM_SEED"]),
        desc="QST sim",
        state_param=state_param,
        n_qubits=n_qubits,
        tensor_bond_dim=tensor_bond_dim,
        tn_avoid_dense=tn_avoid_dense,
        brickwall_ansatz_layers=max(1, int(config.get("BRICKWALL_ANSATZ_LAYERS", 1))),
        brickwall_unified_ry_rx=bool(config.get("BRICKWALL_UNIFIED_RY_RX", False)),
        brickwall_cnot_topology=str(config.get("BRICKWALL_CNOT_TOPOLOGY", "linear")).lower(),
    )
    print(f"  simulate_parallel done in {time.perf_counter() - t_sim:.1f}s")

    data_store = npe_data_device
    t_pack = time.perf_counter()
    n_th, d_th = theta_gpu.shape
    theta_mb = n_th * d_th * 4 / 1e6
    if data_store == "cuda":
        print(f"  Keeping θ on GPU for sbi ({n_th}×{d_th}, ~{theta_mb:.0f} MB)...")
        theta_t = theta_gpu.detach().to(dtype=torch.float32, device="cuda")
        del theta_gpu
        print(f"  θ on cuda in {time.perf_counter() - t_pack:.1f}s")
    else:
        print(f"  Copying θ GPU→CPU for sbi ({n_th}×{d_th}, ~{theta_mb:.0f} MB)...")
        theta_t = theta_gpu.detach().to(dtype=torch.float32, device="cpu")
        del theta_gpu
        if train_on_cuda:
            torch.cuda.empty_cache()
        print(f"  θ on CPU in {time.perf_counter() - t_pack:.1f}s (GPU cache cleared)")

    if standardize_x:
        x_mean_np = x_np.mean(axis=0).astype(np.float64)
        x_std_np = np.maximum(x_np.std(axis=0), 1e-3).astype(np.float64)
        x_np_fit = ((x_np - x_mean_np) / x_std_np).astype(np.float32)
        np.savez(
            os.path.join(results_dir, "x_standardizer.npz"),
            mean=x_mean_np,
            std=x_std_np,
        )
        print(
            f"  STANDARDIZE_X: per-obs z-score from simulations "
            f"(std in [{x_std_np.min():.4g}, {x_std_np.max():.4g}])"
        )
        # Keep embedding_net on CPU; sbi moves it to the training device internally.
        embed = _StandardizeX(
            torch.as_tensor(x_mean_np, dtype=torch.float32, device="cpu"),
            torch.as_tensor(x_std_np, dtype=torch.float32, device="cpu"),
        )
    else:
        x_mean_np = np.zeros(n_obs, dtype=np.float64)
        x_std_np = np.ones(n_obs, dtype=np.float64)
        x_np_fit = x_np
        embed = _PassThroughX()
        print("  STANDARDIZE_X: off (raw frequencies)")

    x_obs_np = x_obs.detach().cpu().numpy().reshape(-1)
    # Pass raw frequencies to sbi.  When STANDARDIZE_X=True, the embedding_net
    # performs the z-score transform inside the density estimator.
    x_for_posterior = torch.as_tensor(x_obs_np.astype(np.float32), dtype=torch.float32, device=device)
    if x_for_posterior.dim() == 1:
        x_for_posterior = x_for_posterior.unsqueeze(0)

    x_t = torch.as_tensor(x_np.astype(np.float32), dtype=torch.float32, device=data_store)

    if bool(config.get("SAVE_NPE_TRAINING_NPZ", True)):
        _train_npz = os.path.join(results_dir, "npe_training_data.npz")
        theta_np = theta_t.detach().cpu().numpy()
        compress = bool(config.get("NPE_TRAINING_NPZ_COMPRESSED", False))
        est_mb = (theta_np.nbytes + x_np.nbytes + x_np_fit.nbytes) / 1e6
        print(
            f"  Saving NPE training data → {_train_npz} "
            f"({'gzip' if compress else 'uncompressed'}, ~{est_mb:.0f} MB raw)..."
        )
        t_save = time.perf_counter()
        payload = dict(theta=theta_np, x_raw=x_np, x_fit=x_np_fit)
        if compress:
            np.savez_compressed(_train_npz, **payload)
        else:
            np.savez(_train_npz, **payload)
        print(
            f"  Saved in {time.perf_counter() - t_save:.1f}s "
            f"(theta {theta_np.shape}, x {x_np.shape})"
        )

    density_estimator = _build_density_estimator(config, n_obs, embed)
    inference = NPE(prior=prior, density_estimator=density_estimator, device=device)

    # Probe on CPU only — instantiating 200M+ params on GPU here can OOM before train().
    _de_tmp = density_estimator(
        torch.zeros(2, num_params, device="cpu"),
        torch.zeros(2, n_obs, device="cpu"),
    )
    n_de_params = sum(p.numel() for p in _de_tmp.parameters())
    print(
        f"  Density estimator parameters: {n_de_params:,} "
        f"(θ_dim={num_params}, train_device={device!r})"
    )
    del _de_tmp
    if train_on_cuda:
        torch.cuda.empty_cache()

    data_mb = (theta_t.numel() + x_t.numel()) * 4 / 1e6
    store_label = "GPU VRAM" if npe_data_device == "cuda" else "CPU RAM"
    print(
        f"  append_simulations: {theta_t.shape[0]} pairs, data_device={npe_data_device!r} "
        f"(~{data_mb:.0f} MB on {store_label}, train on {device})..."
    )
    t_append = time.perf_counter()
    inference.append_simulations(theta_t, x_t, data_device=npe_data_device)
    print(f"  append_simulations done in {time.perf_counter() - t_append:.1f}s")
    print(
        f"Training NPE (NLL) on {device!r} "
        f"(batch={training_batch_size}, pin_memory={train_kwargs['dataloader_kwargs']['pin_memory']})..."
    )
    estimator = inference.train(**train_kwargs)
    if estimator is not None:
        est_dev = next(estimator.parameters()).device
        print(f"  Estimator device after train: {est_dev}")
        if train_on_cuda and est_dev.type != "cuda":
            print("  ERROR: expected CUDA estimator — training may have fallen back to CPU.")

    posterior = _build_posterior(inference, estimator, prior, device, x_for_posterior)

    if estimator is not None:
        torch.save(estimator, os.path.join(results_dir, "estimator.pt"))

    n_post = int(config["NUM_POSTERIOR_SAMPLES"])
    try:
        posterior.prior = PermissivePrior()
    except AttributeError:
        try:
            posterior._prior = PermissivePrior()
        except AttributeError:
            pass

    print(f"\nDrawing {n_post} posterior samples...")
    samples = draw_posterior_samples(posterior, x_for_posterior, n_post)

    return posterior, samples, estimator
