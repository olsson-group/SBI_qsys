"""Training loop for the Trotter residual denoiser."""

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def train_denoiser(
    model: nn.Module,
    train_data: Dict[str, np.ndarray],
    epochs: int = 300,
    lr: float = 1e-4,
    batch_size: int = 64,
    device: str = "cpu",
) -> List[float]:
    """Fit Δ̂ ≈ x_ideal - x_noisy with MSE, Adam and a cosine learning-rate schedule.

    Returns the per-epoch mean training loss.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    def _t(arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32, device=device)

    x_noisy = _t(train_data["noisy"])
    x_ideal = _t(train_data["ideal"])
    n_steps = _t(train_data["n_steps"])

    N = len(x_noisy)
    train_losses: List[float] = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            pred_delta = model.predict_delta(x_noisy[idx], n_steps[idx])
            loss = nn.functional.mse_loss(pred_delta, x_ideal[idx] - x_noisy[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        train_losses.append(epoch_loss / max(1, n_batches))
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch + 1:>4}/{epochs}: train={train_losses[-1]:.6f}")
    return train_losses
