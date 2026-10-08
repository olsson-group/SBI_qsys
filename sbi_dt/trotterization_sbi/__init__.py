"""Trotter denoising as SBI: an NSF posterior p(x_ideal - x_noisy | x_noisy, n_steps), trained with sbi NPE."""

from .training import train_npe
from .evaluation import evaluate_and_plot

__all__ = ["train_npe", "evaluate_and_plot"]
