from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from .base import ModelFitResult
from .validation import compute_r_squared, validate_time_concentration


def langmuir_hinshelwood_concentration(time: np.ndarray, k: float, k_ads: float, c0: float) -> np.ndarray:
    """Langmuir-Hinshelwood concentration profile.

    Differential form:
        dC/dt = -(k * k_ads * C) / (1 + k_ads * C)

    Parameters:
        k     : intrinsic reaction rate constant
        k_ads : adsorption equilibrium constant
        c0    : initial concentration
    """

    t = np.asarray(time, dtype=float)
    if t.size == 0:
        return np.array([], dtype=float)

    def ode(_time: float, y: np.ndarray) -> np.ndarray:
        c = y[0]
        denom = 1.0 + (k_ads * c)
        return np.array([-(k * k_ads * c) / denom], dtype=float)

    solution = solve_ivp(
        ode,
        t_span=(float(t[0]), float(t[-1])),
        y0=np.array([c0], dtype=float),
        t_eval=t,
        method="LSODA",
    )
    if not solution.success:
        raise RuntimeError(f"Langmuir-Hinshelwood integration failed: {solution.message}")
    return solution.y[0]


def fit_langmuir_hinshelwood(
    time: np.ndarray,
    concentration: np.ndarray,
    *,
    initial_guess: Optional[Dict[str, float]] = None,
) -> ModelFitResult:
    """Fit a Langmuir-Hinshelwood model using nonlinear least squares."""

    t, c = validate_time_concentration(time, concentration)

    guess = {
        "k": 0.1,
        "k_ads": 0.1,
        "c0": float(c[0]),
    }
    if initial_guess:
        guess.update(initial_guess)

    x0 = np.array([guess["k"], guess["k_ads"], guess["c0"]], dtype=float)

    def residuals(params: np.ndarray) -> np.ndarray:
        k, k_ads, c0 = params
        pred = langmuir_hinshelwood_concentration(t, k, k_ads, c0)
        return c - pred

    result = least_squares(
        residuals,
        x0,
        bounds=(np.array([0.0, 0.0, -np.inf]), np.array([np.inf, np.inf, np.inf])),
        max_nfev=20000,
    )
    if not result.success:
        raise RuntimeError(f"Langmuir-Hinshelwood fit failed: {result.message}")

    pred = langmuir_hinshelwood_concentration(t, *result.x)
    res = c - pred

    params = {"k": float(result.x[0]), "k_ads": float(result.x[1]), "c0": float(result.x[2])}
    return ModelFitResult(
        model_name="langmuir_hinshelwood",
        parameters=params,
        predictions=pred,
        residuals=res,
        r_squared=compute_r_squared(c, pred),
    )
