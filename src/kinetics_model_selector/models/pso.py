from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.optimize import curve_fit

from .base import ModelFitResult
from .validation import compute_r_squared, validate_time_concentration


def pso_concentration(time: np.ndarray, k2: float, c_eq: float, c0: float) -> np.ndarray:
    """Pseudo-second-order concentration profile.

    Functional form:
        C(t) = C_eq - (C_eq - C0) / (1 + k2 * (C_eq - C0) * t)

    Parameters:
        k2   : pseudo-second-order rate constant
        c_eq : equilibrium concentration
        c0   : initial concentration
    """

    t = np.asarray(time, dtype=float)
    delta = c_eq - c0
    return c_eq - (delta / (1.0 + (k2 * delta * t)))


def fit_pso(
    time: np.ndarray,
    concentration: np.ndarray,
    *,
    initial_guess: Optional[Dict[str, float]] = None,
) -> ModelFitResult:
    """Fit a pseudo-second-order model using nonlinear least squares."""

    t, c = validate_time_concentration(time, concentration)

    guess = {
        "k2": 0.1,
        "c_eq": float(c[-1]),
        "c0": float(c[0]),
    }
    if initial_guess:
        guess.update(initial_guess)

    p0 = [guess["k2"], guess["c_eq"], guess["c0"]]
    bounds = ([0.0, -np.inf, -np.inf], [np.inf, np.inf, np.inf])

    popt, _ = curve_fit(pso_concentration, t, c, p0=p0, bounds=bounds, maxfev=20000)

    prediction = pso_concentration(t, *popt)
    residuals = c - prediction

    params = {"k2": float(popt[0]), "c_eq": float(popt[1]), "c0": float(popt[2])}
    return ModelFitResult(
        model_name="pso",
        parameters=params,
        predictions=prediction,
        residuals=residuals,
        r_squared=compute_r_squared(c, prediction),
    )
