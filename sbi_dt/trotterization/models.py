"""Residual denoiser for Trotter expectation values."""

import torch
import torch.nn as nn

from sbi_dt.tools.models import _mlp


class TrotterResidualDenoiser(nn.Module):
    """x_ideal ≈ x_noisy + Δ̂(x_noisy, n_steps), with Δ̂ an MLP on [x_noisy, log(n_steps + 1)]."""

    def __init__(self, n_obs: int, hidden_dim: int = 256, mlp_num_layers: int = 2):
        super().__init__()
        self.delta_net = _mlp(n_obs + 1, hidden_dim, n_obs, mlp_num_layers)

    def predict_delta(self, x_noisy: torch.Tensor, n_steps: torch.Tensor) -> torch.Tensor:
        feats = torch.cat([x_noisy, torch.log(n_steps + 1.0).unsqueeze(-1)], dim=-1)
        return self.delta_net(feats)

    def forward(self, x_noisy: torch.Tensor, n_steps: torch.Tensor) -> torch.Tensor:
        return x_noisy + self.predict_delta(x_noisy, n_steps)
