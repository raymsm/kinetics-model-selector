"""Final report helpers to integrate model selection and uncertainty outputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from selection.bayesian import BayesianScore
from uncertainty.monte_carlo import MonteCarloSummary


@dataclass(frozen=True)
class ReportTables:
    model_probability_table: list[dict[str, float | str]]
    parameter_uncertainty_table: list[dict[str, float | str]]


def build_report_tables(
    bayesian_scores: list[BayesianScore],
    mc_summary: MonteCarloSummary,
) -> ReportTables:
    """Build table-ready structures for final report rendering."""

    probability_table = [
        {
            "model": score.model_name,
            "criterion": score.criterion,
            "log_evidence": score.log_evidence,
            "posterior_probability": score.posterior_probability,
        }
        for score in bayesian_scores
    ]

    parameter_table: list[dict[str, float | str]] = []
    for model_name, params in mc_summary.parameter_intervals.items():
        for param_name, interval in params.items():
            parameter_table.append(
                {
                    "model": model_name,
                    "parameter": param_name,
                    "median": interval.median,
                    "lower": interval.lower,
                    "upper": interval.upper,
                }
            )

    return ReportTables(
        model_probability_table=probability_table,
        parameter_uncertainty_table=parameter_table,
    )


def add_uncertainty_band_to_plot(
    ax,
    x: np.ndarray,
    prediction_interval: dict[str, np.ndarray],
    *,
    color: str = "tab:blue",
    alpha: float = 0.2,
    label: str = "95% uncertainty band",
) -> None:
    """Optionally render uncertainty bands on fit plots using matplotlib axes."""

    ax.plot(x, prediction_interval["median"], color=color, label=f"{label} median")
    ax.fill_between(
        x,
        prediction_interval["lower"],
        prediction_interval["upper"],
        color=color,
        alpha=alpha,
        label=label,
    )
