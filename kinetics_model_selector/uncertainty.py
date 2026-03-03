from __future__ import annotations

import random

from .mathutils import percentile


def monte_carlo_parameter_samples(
    t: list[float],
    q: list[float],
    fit_fn,
    param_names: list[str],
    n_iter: int = 100,
    noise_std: float = 0.01,
    random_seed: int = 0,
) -> tuple[list[list[float]], list[list[float]]]:
    rng = random.Random(random_seed)
    samples: list[list[float]] = []
    for _ in range(n_iter):
        q_noisy = [qi + rng.gauss(0.0, noise_std) for qi in q]
        fit = fit_fn(t, q_noisy)
        samples.append([fit[name] for name in param_names])

    intervals: list[list[float]] = []
    for col_idx in range(len(param_names)):
        column = [row[col_idx] for row in samples]
        intervals.append([percentile(column, 2.5), percentile(column, 97.5)])
    return samples, intervals
