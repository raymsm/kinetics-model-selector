"""Monte Carlo uncertainty propagation for kinetic models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class RefitModel(Protocol):
    """Protocol expected for Monte Carlo re-fitting.

    The object should be cloneable by calling ``copy()`` and provide
    ``fit(x, y)``, ``predict(x)``, and ``parameters`` (mapping-like).
    """

    name: str

    def copy(self) -> "RefitModel": ...

    def fit(self, x: np.ndarray, y: np.ndarray) -> None: ...

    def predict(self, x: np.ndarray) -> np.ndarray: ...

    @property
    def parameters(self) -> dict[str, float]: ...


@dataclass(frozen=True)
class NoiseModelConfig:
    """Measurement-noise assumptions used to sample synthetic datasets.

    Args:
        kind: "gaussian" (additive) or "relative" (multiplicative).
        sigma: standard deviation term.
        floor: minimum standard deviation for numerical stability.
    """

    kind: str = "gaussian"
    sigma: float = 0.05
    floor: float = 1e-12


@dataclass(frozen=True)
class IntervalSummary:
    """Summary for median and central interval."""

    median: float
    lower: float
    upper: float


@dataclass
class MonteCarloSummary:
    """Aggregated uncertainty outputs for reporting/plotting."""

    parameter_intervals: dict[str, dict[str, IntervalSummary]]
    prediction_intervals: dict[str, dict[str, np.ndarray]]
    successful_refits: dict[str, int]


def sample_synthetic_dataset(
    y_hat: np.ndarray,
    noise: NoiseModelConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a synthetic observed dataset from a noise assumption."""

    if noise.kind == "gaussian":
        scale = np.full_like(y_hat, max(noise.sigma, noise.floor), dtype=float)
        return y_hat + rng.normal(loc=0.0, scale=scale)

    if noise.kind == "relative":
        scale = np.maximum(np.abs(y_hat) * noise.sigma, noise.floor)
        return y_hat + rng.normal(loc=0.0, scale=scale)

    raise ValueError("noise.kind must be one of {'gaussian', 'relative'}")


def _central_interval(samples: np.ndarray, alpha: float) -> IntervalSummary:
    low_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    return IntervalSummary(
        median=float(np.percentile(samples, 50.0)),
        lower=float(np.percentile(samples, low_q)),
        upper=float(np.percentile(samples, hi_q)),
    )


def propagate_monte_carlo_uncertainty(
    models: list[RefitModel],
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_samples: int = 200,
    noise: NoiseModelConfig | None = None,
    ci_level: float = 0.95,
    random_seed: int | None = None,
) -> MonteCarloSummary:
    """Refit models over synthetic datasets and summarize uncertainty.

    Workflow:
      1) fit each model to original data and generate y-hat,
      2) sample synthetic datasets from noise assumptions,
      3) refit models on each sample,
      4) compute central intervals for parameters and predictions.
    """

    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")

    noise = noise or NoiseModelConfig()
    alpha = 1.0 - ci_level
    if not (0.0 < alpha < 1.0):
        raise ValueError("ci_level must be in (0, 1)")

    rng = np.random.default_rng(seed=random_seed)

    parameter_intervals: dict[str, dict[str, IntervalSummary]] = {}
    prediction_intervals: dict[str, dict[str, np.ndarray]] = {}
    successful_refits: dict[str, int] = {}

    for model in models:
        baseline = model.copy()
        baseline.fit(x, y)
        y_hat = baseline.predict(x)

        parameter_samples: dict[str, list[float]] = {}
        prediction_samples: list[np.ndarray] = []
        success_count = 0

        for _ in range(n_samples):
            y_syn = sample_synthetic_dataset(y_hat=y_hat, noise=noise, rng=rng)
            candidate = model.copy()
            try:
                candidate.fit(x, y_syn)
                pred = candidate.predict(x)
                prediction_samples.append(np.asarray(pred, dtype=float))
                for name, value in candidate.parameters.items():
                    parameter_samples.setdefault(name, []).append(float(value))
                success_count += 1
            except Exception:
                continue

        successful_refits[model.name] = success_count
        if success_count < 2:
            parameter_intervals[model.name] = {}
            prediction_intervals[model.name] = {
                "median": np.asarray(y_hat, dtype=float),
                "lower": np.asarray(y_hat, dtype=float),
                "upper": np.asarray(y_hat, dtype=float),
            }
            continue

        param_summary = {
            param_name: _central_interval(np.asarray(values, dtype=float), alpha=alpha)
            for param_name, values in parameter_samples.items()
            if len(values) >= 2
        }
        parameter_intervals[model.name] = param_summary

        pred_arr = np.asarray(prediction_samples, dtype=float)
        prediction_intervals[model.name] = {
            "median": np.percentile(pred_arr, 50.0, axis=0),
            "lower": np.percentile(pred_arr, 100.0 * alpha / 2.0, axis=0),
            "upper": np.percentile(pred_arr, 100.0 * (1.0 - alpha / 2.0), axis=0),
        }

    return MonteCarloSummary(
        parameter_intervals=parameter_intervals,
        prediction_intervals=prediction_intervals,
        successful_refits=successful_refits,
    )
