"""NPE with a neural spline flow (sbi) for the Trotter residual Δx = x_ideal - x_noisy.

theta = Δx                          (n_obs,)
x     = [x_noisy | log(n_steps+1)]  (n_obs + 1,)
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
from sbi.inference import NPE
from sbi.inference.posteriors import DirectPosterior
from sbi.neural_nets import posterior_nn
from torch.distributions import Independent, Normal


def train_npe(
    train_data: Dict[str, np.ndarray],
    config: dict,
    device: str = "cpu",
) -> Tuple[object, object, List[float], List[float]]:
    """Train the posterior q(Δx | x_noisy, n_steps).

    Returns (posterior, estimator, train_log_probs, val_log_probs); sample with
    ``posterior.sample((K,), x=x_obs)`` and save the estimator with ``torch.save``.
    """
    theta = torch.tensor(train_data["ideal"], dtype=torch.float32) - torch.tensor(train_data["noisy"], dtype=torch.float32)
    x = torch.cat([
        torch.tensor(train_data["noisy"], dtype=torch.float32),
        torch.log(torch.tensor(train_data["n_steps"], dtype=torch.float32) + 1.0).unsqueeze(-1),
    ], dim=-1)

    # Diagonal Gaussian prior fitted to the training Δx
    prior = Independent(Normal(theta.mean(dim=0).to(device), theta.std(dim=0).clamp(min=0.05).to(device)), 1)

    density_estimator_fn = posterior_nn(
        model="zuko_nsf",
        hidden_features=config.get("flow_hidden_features", 256),
        num_transforms=config.get("num_transforms", 5),
        num_bins=config.get("nsf_num_bins", 8),
    )
    n_params = sum(p.numel() for p in density_estimator_fn(theta[:2], x[:2]).parameters())
    print(f"  NSF parameters: {n_params:,}")

    inference = NPE(prior=prior, density_estimator=density_estimator_fn, device=device)
    inference.append_simulations(theta, x)
    estimator = inference.train(
        training_batch_size=config.get("batch_size", 256),
        learning_rate=config.get("lr", 1e-3),
        max_num_epochs=config.get("epochs", 300),
        stop_after_epochs=config.get("stop_after_epochs", 50),
        show_train_summary=True,
        clip_max_norm=config.get("clip_max_norm", 5.0),
    )
    train_log_probs = list(inference._summary.get("training_log_probs", []))
    val_log_probs = list(inference._summary.get("validation_log_probs", []))
    posterior = DirectPosterior(posterior_estimator=estimator, prior=prior, device=device)
    return posterior, estimator, train_log_probs, val_log_probs
