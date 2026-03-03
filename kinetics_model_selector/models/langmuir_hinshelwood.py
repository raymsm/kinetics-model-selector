from __future__ import annotations

from ..mathutils import linear_regression, sse


def predict(t: list[float], k: float, K: float) -> list[float]:
    return [(k * ti) / (1.0 + K * ti) for ti in t]


def fit(t: list[float], q: list[float]) -> dict[str, float]:
    x = [1.0 / ti for ti, qi in zip(t, q) if ti > 1e-12 and qi > 1e-12]
    y = [1.0 / qi for ti, qi in zip(t, q) if ti > 1e-12 and qi > 1e-12]

    slope, intercept = linear_regression(x, y)
    k = 1.0 / max(slope, 1e-12)
    K = max(intercept * k, 1e-12)
    pred = predict(t, k, K)
    return {"sse": sse(q, pred), "k": k, "K": K}
