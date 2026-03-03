from __future__ import annotations

from .models import langmuir_hinshelwood, pfo, pso

MODELS = {
    "pfo": (pfo.fit, pfo.predict),
    "pso": (pso.fit, pso.predict),
    "langmuir_hinshelwood": (langmuir_hinshelwood.fit, langmuir_hinshelwood.predict),
}


def fit_models(t: list[float], q: list[float]) -> dict[str, dict[str, float]]:
    return {name: fit_fn(t, q) for name, (fit_fn, _) in MODELS.items()}


def select_best_model(t: list[float], q: list[float]) -> tuple[str, dict[str, float]]:
    results = fit_models(t, q)
    best_name = min(results, key=lambda key: results[key]["sse"])
    return best_name, results[best_name]
