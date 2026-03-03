from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.optimize import curve_fit

from .base import ModelFitResult
from .validation import compute_r_squared, validate_time_concentration


def pfo_concentration(time: np.ndarray, k1: float, c_eq: float, c0: float) -> np.ndarray:
    """Pseudo-first-order concentration profile.

    Functional form:
        C(t) = C_eq - (C_eq - C0) * exp(-k1 * t)

    Parameters:
        k1   : pseudo-first-order rate constant
        c_eq : equilibrium concentration
        c0   : initial concentration
    """

    t = np.asarray(time, dtype=float)
    return c_eq - (c_eq - c0) * np.exp(-k1 * t)


def fit_pfo(
    time: np.ndarray,
    concentration: np.ndarray,
    *,
    initial_guess: Optional[Dict[str, float]] = None,
) -> ModelFitResult:
    """Fit a pseudo-first-order model using nonlinear least squares."""

    t, c = validate_time_concentration(time, concentration)

    guess = {
        "k1": 0.1,
        "c_eq": float(c[-1]),
        "c0": float(c[0]),
    }
    if initial_guess:
        guess.update(initial_guess)

    p0 = [guess["k1"], guess["c_eq"], guess["c0"]]
    bounds = ([0.0, -np.inf, -np.inf], [np.inf, np.inf, np.inf])

    popt, _ = curve_fit(pfo_concentration, t, c, p0=p0, bounds=bounds, maxfev=20000)

    prediction = pfo_concentration(t, *popt)
    residuals = c - prediction

    params = {"k1": float(popt[0]), "c_eq": float(popt[1]), "c0": float(popt[2])}
    return ModelFitResult(
        model_name="pfo",
        parameters=params,
        predictions=prediction,
        residuals=residuals,
        r_squared=compute_r_squared(c, prediction),
    )
