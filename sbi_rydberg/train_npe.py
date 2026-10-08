"""Amortised NPE: θ ~ BoxUniform -> simulator -> x, conditional flow q(θ | x)."""

from __future__ import annotations

import os
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform


class _SummaryNet(nn.Module):
    """Standardize x, then compress n_obs -> embed_dim with a small MLP."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, n_obs: int, embed_dim: int):
        super().__init__()
        self.register_buffer("mean", mean.float().reshape(1, -1).clone())
        self.register_buffer("std", torch.clamp(std.float().reshape(1, -1).clone(), min=1e-6))
        hidden = max(embed_dim * 2, 256)
        self.mlp = nn.Sequential(
            nn.Linear(n_obs, hidden), nn.SiLU(),
            nn.Linear(hidden, embed_dim), nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.mlp((x - self.mean) / self.std)


def train_npe_amortised(
    config: dict,
    x_obs: torch.Tensor,
    results_dir: str,
    *,
    num_params: int,
    prior_low: float,
    prior_high: float,
    simulate_batch: Callable[[np.ndarray], np.ndarray],
) -> tuple[object, np.ndarray]:
    """Simulate training data, train the flow, return (posterior, posterior samples at x_obs).

    Saves train_theta/train_x, x_standardizer, npe_training_data and estimator.pt to results_dir.
    """
    device = str(config["DEVICE"])
    n_sims = int(config["NPE_N_SIMS"])

    prior = BoxUniform(
        low=torch.full((num_params,), float(prior_low), device=device),
        high=torch.full((num_params,), float(prior_high), device=device),
    )

    print(f"\nDrawing {n_sims} prior samples ({num_params}D θ) and running simulator...")
    theta = prior.sample((n_sims,)).cpu().numpy().astype(np.float32)
    x_np = simulate_batch(theta)
    if x_np.ndim != 2 or x_np.shape[0] != n_sims:
        raise ValueError(f"simulate_batch must return (N, n_obs); got {x_np.shape}")
    np.save(os.path.join(results_dir, "train_theta.npy"), theta)
    np.save(os.path.join(results_dir, "train_x.npy"), x_np)

    x_mean = x_np.mean(axis=0).astype(np.float64)
    x_std = np.maximum(x_np.std(axis=0), 1e-6).astype(np.float64)
    np.savez(os.path.join(results_dir, "x_standardizer.npz"), mean=x_mean, std=x_std)
    np.savez_compressed(
        os.path.join(results_dir, "npe_training_data.npz"),
        theta=theta,
        x_raw=x_np,
        x_fit=((x_np - x_mean) / x_std).astype(np.float32),
    )

    x_obs_np = x_obs.detach().cpu().numpy().reshape(-1)
    if x_obs_np.size != x_np.shape[1]:
        raise ValueError(f"x_obs length {x_obs_np.size} != simulator n_obs {x_np.shape[1]}")

    embed = _SummaryNet(
        torch.as_tensor(x_mean, dtype=torch.float32, device=device),
        torch.as_tensor(x_std, dtype=torch.float32, device=device),
        n_obs=x_np.shape[1],
        embed_dim=int(config["EMBED_DIM"]),
    )
    density_estimator = posterior_nn(
        model=config["FLOW_MODEL"],
        hidden_features=int(config["FLOW_HIDDEN_FEATURES"]),
        num_transforms=int(config["NUM_TRANSFORMS"]),
        num_bins=int(config["NSF_NUM_BINS"]),
        embedding_net=embed,
    )

    inference = NPE(prior=prior, density_estimator=density_estimator, device=device)
    inference.append_simulations(
        torch.as_tensor(theta, dtype=torch.float32, device=device),
        torch.as_tensor(x_np, dtype=torch.float32, device=device),
    )
    print("Training NPE (NLL)...")
    estimator = inference.train(
        training_batch_size=int(config["BATCH_SIZE"]),
        learning_rate=float(config["LR"]),
        max_num_epochs=int(config["EPOCHS"]),
        stop_after_epochs=int(config["STOP_AFTER_EPOCHS"]),
        clip_max_norm=float(config["CLIP_MAX_NORM"]),
        show_train_summary=True,
    )
    torch.save(estimator, os.path.join(results_dir, "estimator.pt"))

    x_cond = torch.as_tensor(x_obs_np, dtype=torch.float32, device=device).reshape(1, -1)
    posterior = inference.build_posterior(estimator, prior=prior, sample_with="direct")
    posterior = posterior.set_default_x(x_cond)

    n_post = int(config["NUM_POSTERIOR_SAMPLES"])
    print(f"\nDrawing {n_post} posterior samples...")
    samples = posterior.sample((n_post,), x=x_cond, show_progress_bars=False)
    return posterior, samples.detach().cpu().numpy().astype(np.float32)
