from __future__ import annotations

from typing import Tuple

import numpy as np


MIN_SAMPLES = 3


def validate_time_concentration(
    time: np.ndarray,
    concentration: np.ndarray,
    *,
    min_samples: int = MIN_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate time/concentration arrays and return normalized float arrays."""

    t = np.asarray(time, dtype=float)
    c = np.asarray(concentration, dtype=float)

    if t.ndim != 1 or c.ndim != 1:
        raise ValueError("time and concentration must be 1D arrays")
    if t.size != c.size:
        raise ValueError("time and concentration must have the same length")
    if t.size < min_samples:
        raise ValueError(f"at least {min_samples} samples are required")
    if np.isnan(t).any() or np.isnan(c).any():
        raise ValueError("time and concentration cannot contain NaN values")
    if np.isinf(t).any() or np.isinf(c).any():
        raise ValueError("time and concentration cannot contain infinite values")
    if np.any(np.diff(t) <= 0):
        raise ValueError("time must be strictly increasing (monotonic)")

    return t, c


def compute_r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Compute R² with a stable fallback for zero-variance observations."""

    ss_res = float(np.sum((observed - predicted) ** 2))
    ss_tot = float(np.sum((observed - np.mean(observed)) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - (ss_res / ss_tot)
