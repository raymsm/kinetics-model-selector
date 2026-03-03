from __future__ import annotations

from ..mathutils import linear_regression, sse


def predict(t: list[float], qe: float, k2: float) -> list[float]:
    return [(k2 * (qe**2) * ti) / (1.0 + k2 * qe * ti) for ti in t]


def fit(t: list[float], q: list[float]) -> dict[str, float]:
    x = [ti for ti, qi in zip(t, q) if qi > 1e-12]
    y = [ti / qi for ti, qi in zip(t, q) if qi > 1e-12]

    slope, intercept = linear_regression(x, y)
    qe = 1.0 / max(slope, 1e-12)
    k2 = 1.0 / max(intercept * (qe**2), 1e-12)
    pred = predict(t, qe, k2)
    return {"sse": sse(q, pred), "qe": qe, "k2": k2}
