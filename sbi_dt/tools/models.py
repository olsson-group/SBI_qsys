"""Shared model building blocks."""

import torch.nn as nn


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, num_layers: int) -> nn.Sequential:
    """Build MLP with `num_layers` hidden layers and GELU activations.
    num_layers=1: Linear(in,h) → GELU → Linear(h,out)
    num_layers=2: Linear(in,h) → GELU → Linear(h,h) → GELU → Linear(h,out)  (default)
    """
    layers: list = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
    for _ in range(num_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)
