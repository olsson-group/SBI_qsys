"""Zero-noise extrapolation of ⟨O⟩ measured at several noise gains."""

from typing import Optional, Tuple

import numpy as np


def extrapolate_to_zero(
    gains: np.ndarray,
    expectations: np.ndarray,
    method: str = "exponential",
) -> Tuple[float, dict]:
    """Extrapolate ⟨O⟩ to zero noise (G = 0).

    method: "linear", "quadratic", or "exponential" (log-linear fit y = A·exp(B·G); falls back to
    linear when the values change sign or vanish).

    Returns (estimate at G=0, fit_info) with fit_info = {"method", "params", optional "fallback"}.
    """
    gains = np.asarray(gains, dtype=float)
    expectations = np.asarray(expectations, dtype=float)
    if len(gains) != len(expectations) or len(gains) < 2:
        raise ValueError("Need at least 2 (gain, expectation) pairs")

    fit_info: dict = {}

    def _exponential() -> Optional[float]:
        signs = np.sign(expectations)
        if not (np.all(signs == signs[0]) and np.all(np.abs(expectations) > 1e-12)):
            return None
        B, log_A = np.polyfit(gains, np.log(np.abs(expectations)), 1)
        A = signs[0] * np.exp(log_A)
        fit_info["method"] = "exponential"
        fit_info["params"] = {"a": float(A), "b": float(B), "c": 0.0}   # y = a·exp(b·G) + c
        return float(A)

    def _linear() -> float:
        a, b = np.polyfit(gains, expectations, 1)
        fit_info["method"] = "linear"
        fit_info["params"] = {"slope": float(a), "intercept": float(b)}
        return b

    def _quadratic() -> float:
        a, b, c = np.polyfit(gains, expectations, 2)
        fit_info["method"] = "quadratic"
        fit_info["params"] = {"a": float(a), "b": float(b), "c": float(c)}
        return c

    if method == "exponential":
        val = _exponential()
        if val is None:
            val = _linear()
            fit_info["fallback"] = "exponential failed, used linear"
    elif method == "linear":
        val = _linear()
    elif method == "quadratic":
        val = _quadratic()
    else:
        raise ValueError(f"method must be 'linear', 'quadratic' or 'exponential', got {method!r}")
    return float(val), fit_info
