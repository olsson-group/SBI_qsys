"""Probabilistic error amplification (PEA) for ZNE.

Noise gain G scales the Pauli noise rates λ_i by G. The SBI parameters are channel probabilities p_i:
    p_i = (1 - exp(-2λ_i)) / 2   <=>   λ_i = -ln(1 - 2p_i) / 2
so p_i → λ_i is converted, scaled by G, and converted back before sampling Pauli insertions.
"""

import numpy as np

from sbi_pec_zne.common import PAULI_2Q


def _pauli_probs(rates: np.ndarray) -> np.ndarray:
    """16 probabilities [II, IX, ..., ZZ] from the 15 non-identity rates (remainder goes to II)."""
    rates = np.clip(rates, 0.0, 1.0)
    s = np.sum(rates)
    if s >= 1.0:
        return np.concatenate(([0.0], rates / s))
    return np.concatenate(([1.0 - s], rates))


def sample_pauli_errors_for_circuit(rates_matrix: np.ndarray, rng: np.random.Generator) -> tuple:
    """One 2-qubit Pauli per CNOT, drawn from its error probabilities (rates_matrix is (n_cnots, 15))."""
    return tuple(PAULI_2Q[rng.choice(16, p=_pauli_probs(rates))] for rates in rates_matrix)


def sample_pauli_configs_batch(rates_matrix: np.ndarray, n_instances: int, rng: np.random.Generator) -> list:
    """``n_instances`` independent Pauli insertion configurations."""
    return [sample_pauli_errors_for_circuit(rates_matrix, rng) for _ in range(n_instances)]


def _probs_to_lambda(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-12, 0.5 - 1e-9)   # keep log(1 - 2p) finite
    return -np.log(1.0 - 2.0 * p) / 2.0


def _lambda_to_probs(lam: np.ndarray) -> np.ndarray:
    return (1.0 - np.exp(-2.0 * np.maximum(lam, 0.0))) / 2.0


def amplify_rates(rates_matrix: np.ndarray, gain: float) -> np.ndarray:
    """Pauli channel probabilities (n_cnots, 15) with the noise rates λ scaled by ``gain``."""
    p_amp = _lambda_to_probs(_probs_to_lambda(rates_matrix) * gain)
    return np.clip(p_amp, 0.0, 1.0)
