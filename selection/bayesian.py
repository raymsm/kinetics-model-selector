"""Bayesian-style model selection helpers.

This module provides a lightweight evidence approximation workflow for
candidate kinetic models (e.g., PFO/PSO/L-H) and converts approximated
log-evidences into normalized posterior model probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol

import numpy as np


class BayesianModel(Protocol):
    """Protocol for candidate models used in Bayesian approximation.

    Implementers should expose:
      * ``name``: model label (e.g., "PFO", "PSO", "L-H")
      * ``n_parameters``: number of fitted parameters
      * ``n_observations``: number of observed datapoints
      * ``log_likelihood``: total log-likelihood at fitted parameter values
      * ``pointwise_log_likelihood`` (optional): per-observation log-likelihood
        samples with shape ``(n_posterior_samples, n_observations)`` for WAIC/LOO.
    """

    name: str
    n_parameters: int
    n_observations: int
    log_likelihood: float


@dataclass(frozen=True)
class BayesianScore:
    """Model-level Bayesian approximation outputs."""

    model_name: str
    criterion: str
    log_evidence: float
    posterior_probability: float


@dataclass(frozen=True)
class WaicResult:
    """WAIC proxy result."""

    waic: float
    lppd: float
    p_waic: float


@dataclass(frozen=True)
class LooProxyResult:
    """Simple PSIS-LOO-like proxy using log pointwise predictive density."""

    elpd_loo_proxy: float
    looic_proxy: float


def bic_log_evidence(model: BayesianModel) -> float:
    """Approximate log evidence using BIC.

    ``log p(y|M) ≈ log L_hat - 0.5 * k * log(n)``.
    """

    n_obs = max(model.n_observations, 1)
    return float(model.log_likelihood - 0.5 * model.n_parameters * np.log(n_obs))


def waic_from_log_likelihood_samples(log_lik_samples: np.ndarray) -> WaicResult:
    """Compute WAIC from posterior log-likelihood samples.

    Args:
        log_lik_samples: Array ``(n_samples, n_observations)``.
    """

    if log_lik_samples.ndim != 2:
        raise ValueError("log_lik_samples must be 2D: (n_samples, n_observations)")

    # lppd_j = log( E_s[p(y_j|theta_s)] ) = logsumexp(log_lik_sj) - log(S)
    s = log_lik_samples.shape[0]
    max_ll = np.max(log_lik_samples, axis=0)
    stable = log_lik_samples - max_ll
    lppd_per_obs = max_ll + np.log(np.mean(np.exp(stable), axis=0))
    lppd = float(np.sum(lppd_per_obs))

    p_waic = float(np.sum(np.var(log_lik_samples, axis=0, ddof=1)))
    waic = float(-2.0 * (lppd - p_waic))
    return WaicResult(waic=waic, lppd=lppd, p_waic=p_waic)


def loo_proxy_from_log_likelihood_samples(log_lik_samples: np.ndarray) -> LooProxyResult:
    """Compute a lightweight LOO proxy from posterior samples.

    This is a pragmatic approximation using harmonic-mean-like importance
    weighting at the pointwise level. For production use, replace with PSIS-LOO.
    """

    if log_lik_samples.ndim != 2:
        raise ValueError("log_lik_samples must be 2D: (n_samples, n_observations)")

    # approximate elpd_loo_j ≈ -log(mean(exp(-log_lik_sj)))
    neg = -log_lik_samples
    max_neg = np.max(neg, axis=0)
    stable = neg - max_neg
    elpd_per_obs = -(max_neg + np.log(np.mean(np.exp(stable), axis=0)))
    elpd = float(np.sum(elpd_per_obs))
    looic = float(-2.0 * elpd)
    return LooProxyResult(elpd_loo_proxy=elpd, looic_proxy=looic)


def posterior_model_probabilities(
    models: Iterable[BayesianModel],
    method: str = "bic",
    prior_model_probabilities: Mapping[str, float] | None = None,
    pointwise_log_likelihood_samples: Mapping[str, np.ndarray] | None = None,
) -> list[BayesianScore]:
    """Return normalized posterior model probabilities over candidate models.

    Args:
        models: Candidate models, e.g. PFO/PSO/L-H.
        method: ``bic`` (default), ``waic``, or ``loo_proxy``.
        prior_model_probabilities: Optional prior probability for each model.
        pointwise_log_likelihood_samples: Required for WAIC/LOO methods.
    """

    model_list = list(models)
    if not model_list:
        return []

    if prior_model_probabilities is None:
        prior_model_probabilities = {m.name: 1.0 / len(model_list) for m in model_list}

    log_scores: dict[str, float] = {}
    for model in model_list:
        log_prior = float(np.log(prior_model_probabilities.get(model.name, 1e-12)))

        if method == "bic":
            log_evidence = bic_log_evidence(model)
        elif method == "waic":
            if pointwise_log_likelihood_samples is None or model.name not in pointwise_log_likelihood_samples:
                raise ValueError(f"Missing pointwise log-likelihood samples for model '{model.name}'")
            waic = waic_from_log_likelihood_samples(pointwise_log_likelihood_samples[model.name])
            # WAIC = -2 * approx log predictive density => log_evidence proxy = -WAIC/2
            log_evidence = -0.5 * waic.waic
        elif method == "loo_proxy":
            if pointwise_log_likelihood_samples is None or model.name not in pointwise_log_likelihood_samples:
                raise ValueError(f"Missing pointwise log-likelihood samples for model '{model.name}'")
            loo = loo_proxy_from_log_likelihood_samples(pointwise_log_likelihood_samples[model.name])
            log_evidence = loo.elpd_loo_proxy
        else:
            raise ValueError("method must be one of {'bic', 'waic', 'loo_proxy'}")

        log_scores[model.name] = log_evidence + log_prior

    max_score = max(log_scores.values())
    normalizer = sum(np.exp(v - max_score) for v in log_scores.values())

    scores: list[BayesianScore] = []
    for model in model_list:
        joint_log = log_scores[model.name]
        posterior = float(np.exp(joint_log - max_score) / normalizer)
        scores.append(
            BayesianScore(
                model_name=model.name,
                criterion=method,
                log_evidence=float(joint_log),
                posterior_probability=posterior,
            )
        )

    return sorted(scores, key=lambda s: s.posterior_probability, reverse=True)
