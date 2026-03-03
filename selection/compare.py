"""Model comparison utilities for adsorption kinetics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, Literal

ModelLabel = Literal["PFO", "PSO", "L-H"]
Criterion = Literal["r2", "aic", "bic"]


@dataclass
class FitMetrics:
    model: ModelLabel
    r2: float
    aic: float | None
    bic: float | None
    rate_constants: Dict[str, float]
    predictions: list[float]
    residuals: list[float]
    sse: float


@dataclass
class ComparisonResult:
    best_model: ModelLabel
    criterion: Criterion
    metrics: Dict[ModelLabel, FitMetrics]

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "best_model": self.best_model,
            "criterion": self.criterion,
            "models": {},
        }
        for label, metric in self.metrics.items():
            payload["models"][label] = {
                "r2": metric.r2,
                "aic": metric.aic,
                "bic": metric.bic,
                "rate_constants": metric.rate_constants,
                "predictions": metric.predictions,
                "residuals": metric.residuals,
                "sse": metric.sse,
            }
        return payload


def _validate_data(time: Iterable[float], uptake: Iterable[float]) -> tuple[list[float], list[float]]:
    t = [float(v) for v in time]
    q = [float(v) for v in uptake]
    if len(t) != len(q):
        raise ValueError("time and uptake must have matching lengths")
    if len(t) < 3:
        raise ValueError("at least 3 observations are required")
    ordered = sorted(zip(t, q), key=lambda p: p[0])
    return [p[0] for p in ordered], [p[1] for p in ordered]


def _sum(values: Iterable[float]) -> float:
    total = 0.0
    for value in values:
        total += value
    return total


def _polyfit_slope_intercept(x: list[float], y: list[float]) -> tuple[float, float]:
    n = len(x)
    sx = _sum(x)
    sy = _sum(y)
    sxx = _sum(v * v for v in x)
    sxy = _sum(a * b for a, b in zip(x, y))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-15:
        return 0.0, sy / n
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return slope, intercept


def _r2(y_true: list[float], y_pred: list[float]) -> float:
    mean_true = _sum(y_true) / len(y_true)
    ss_res = _sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    ss_tot = _sum((a - mean_true) ** 2 for a in y_true)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1 - ss_res / ss_tot


def _aic_bic(n: int, sse: float, k: int) -> tuple[float, float]:
    safe_sse = max(sse, 1e-15)
    aic = n * math.log(safe_sse / n) + 2 * k
    bic = n * math.log(safe_sse / n) + k * math.log(n)
    return aic, bic


def _fit_pfo(t: list[float], q: list[float]) -> FitMetrics:
    qe = max(q) * 1.001 + 1e-9
    y = [math.log(max(1e-12, 1 - (qv / qe))) for qv in q]
    slope, _ = _polyfit_slope_intercept(t, y)
    k1 = max(1e-12, -slope)
    pred = [qe * (1 - math.exp(-k1 * tv)) for tv in t]
    residuals = [obs - est for obs, est in zip(q, pred)]
    sse = _sum(r * r for r in residuals)
    aic, bic = _aic_bic(len(t), sse, 1)
    return FitMetrics("PFO", _r2(q, pred), aic, bic, {"k1": k1}, pred, residuals, sse)


def _fit_pso(t: list[float], q: list[float]) -> FitMetrics:
    safe_q = [max(1e-12, qv) for qv in q]
    y = [tv / qv for tv, qv in zip(t, safe_q)]
    slope, intercept = _polyfit_slope_intercept(t, y)
    qe = (1 / slope) if abs(slope) > 1e-15 else max(q)
    qe = max(1e-12, qe)
    k2 = (1 / (intercept * qe * qe)) if intercept > 1e-15 else 1e-12
    k2 = max(1e-12, k2)
    pred = [(k2 * qe * qe * tv) / (1 + (k2 * qe * tv)) for tv in t]
    residuals = [obs - est for obs, est in zip(q, pred)]
    sse = _sum(r * r for r in residuals)
    aic, bic = _aic_bic(len(t), sse, 2)
    return FitMetrics("PSO", _r2(q, pred), aic, bic, {"k2": k2}, pred, residuals, sse)


def _fit_lh(t: list[float], q: list[float], grid_size: int = 200) -> FitMetrics:
    best: tuple[float, float, float, list[float], list[float]] | None = None
    log_min = math.log10(1e-5)
    log_max = math.log10(1e2)

    for i in range(grid_size):
        frac = i / (grid_size - 1)
        k_lh = 10 ** (log_min + frac * (log_max - log_min))
        basis = [(k_lh * max(tv, 1e-12)) / (1 + k_lh * max(tv, 1e-12)) for tv in t]
        denom = _sum(v * v for v in basis)
        if denom <= 0:
            continue
        q_max = max(1e-12, _sum(b * qv for b, qv in zip(basis, q)) / denom)
        pred = [q_max * b for b in basis]
        residuals = [obs - est for obs, est in zip(q, pred)]
        sse = _sum(r * r for r in residuals)
        if best is None or sse < best[0]:
            best = (sse, k_lh, q_max, pred, residuals)

    if best is None:
        raise RuntimeError("failed to fit L-H model")

    sse, k_lh, q_max, pred, residuals = best
    aic, bic = _aic_bic(len(t), sse, 2)
    return FitMetrics("L-H", _r2(q, pred), aic, bic, {"k_lh": k_lh, "q_max": q_max}, pred, residuals, sse)


def compare_models(
    time: Iterable[float],
    uptake: Iterable[float],
    *,
    criterion: Criterion = "r2",
    include_info_criteria: bool = True,
) -> ComparisonResult:
    """Fit all supported models on the same input dataset and select the best model."""
    t, q = _validate_data(time, uptake)

    metrics: Dict[ModelLabel, FitMetrics] = {
        "PFO": _fit_pfo(t, q),
        "PSO": _fit_pso(t, q),
        "L-H": _fit_lh(t, q),
    }

    if not include_info_criteria:
        for metric in metrics.values():
            metric.aic = None
            metric.bic = None

    if criterion == "r2":
        best_model = max(metrics.items(), key=lambda pair: pair[1].r2)[0]
    elif criterion == "aic":
        best_model = min(metrics.items(), key=lambda pair: pair[1].aic if pair[1].aic is not None else float("inf"))[0]
    elif criterion == "bic":
        best_model = min(metrics.items(), key=lambda pair: pair[1].bic if pair[1].bic is not None else float("inf"))[0]
    else:
        raise ValueError(f"Unsupported criterion: {criterion}")

    return ComparisonResult(best_model=best_model, criterion=criterion, metrics=metrics)


def comparison_to_rows(result: ComparisonResult) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for label, metric in result.metrics.items():
        row: Dict[str, Any] = {
            "model": label,
            "best_model": result.best_model,
            "criterion": result.criterion,
            "r2": metric.r2,
            "aic": metric.aic,
            "bic": metric.bic,
            "sse": metric.sse,
        }
        for key, value in metric.rate_constants.items():
            row[f"rate_constant_{key}"] = value
        rows.append(row)
    return rows
