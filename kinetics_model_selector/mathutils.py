from __future__ import annotations

import math
import random


def linspace(start: float, stop: float, num: int) -> list[float]:
    if num <= 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


def sse(y_true: list[float], y_pred: list[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred))


def linear_regression(x: list[float], y: list[float]) -> tuple[float, float]:
    n = len(x)
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    den = sum((xi - x_mean) ** 2 for xi in x)
    slope = num / den
    intercept = y_mean - slope * x_mean
    return slope, intercept


def add_gaussian_noise(values: list[float], std: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [v + rng.gauss(0.0, std) for v in values]


def percentile(values: list[float], p: float) -> float:
    sorted_vals = sorted(values)
    if not sorted_vals:
        return math.nan
    k = (len(sorted_vals) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)
