from __future__ import annotations

import math

from ..mathutils import linear_regression, linspace, sse


def predict(t: list[float], qe: float, k1: float) -> list[float]:
    return [qe * (1.0 - math.exp(-k1 * ti)) for ti in t]


def fit(t: list[float], q: list[float]) -> dict[str, float]:
    qe_candidates = linspace(max(q) * 1.01, max(q) * 2.5, 250)
    best = {"sse": float("inf"), "qe": 0.0, "k1": 0.0}

    for qe in qe_candidates:
        y = [math.log(max(qe - qi, 1e-12)) for qi in q]
        slope, _ = linear_regression(t, y)
        k1 = max(1e-12, -slope)
        pred = predict(t, qe, k1)
        err = sse(q, pred)
        if err < best["sse"]:
            best = {"sse": err, "qe": qe, "k1": k1}

    return best
