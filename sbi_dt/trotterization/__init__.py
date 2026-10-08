"""Trotter dynamics: noisy TFIM time evolution, Clifford training data, residual denoiser, ZNE baseline."""

from .circuit import (
    get_tfim_observables,
    build_neel_trotter_circuit,
    build_clifford_trotter_circuit,
    compute_ideal_expectations,
)
from .runner import TrotterNoisyRunner
from .models import TrotterResidualDenoiser
from .data import generate_trotter_data
from .training import train_denoiser
from .evaluation import evaluate_and_plot

__all__ = [
    "get_tfim_observables",
    "build_neel_trotter_circuit",
    "build_clifford_trotter_circuit",
    "compute_ideal_expectations",
    "TrotterNoisyRunner",
    "TrotterResidualDenoiser",
    "generate_trotter_data",
    "train_denoiser",
    "evaluate_and_plot",
]
