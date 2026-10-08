"""Probabilistic error cancellation (PEC) with SBI-inferred noise.

Run from the repository root:  python sbi_pec_zne/pec/run.py

Submodules: sampling (inverse channels, quasi-probability sampling), correct (run_pec_experiment),
visualize (convergence plot).
"""

from .sampling import compute_inverse_pauli_channel, sample_pauli_combinations, run_pec_sampling
from .correct import run_pec_experiment
from .visualize import load_and_plot

__all__ = [
    "compute_inverse_pauli_channel",
    "sample_pauli_combinations",
    "run_pec_sampling",
    "run_pec_experiment",
    "load_and_plot",
]
