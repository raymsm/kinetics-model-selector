"""Configuration and CLI flags for Bayesian and uncertainty assumptions."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass

from uncertainty.monte_carlo import NoiseModelConfig


@dataclass(frozen=True)
class BayesianConfig:
    method: str = "bic"
    pfo_prior: float = 1 / 3
    pso_prior: float = 1 / 3
    lh_prior: float = 1 / 3


@dataclass(frozen=True)
class UncertaintyConfig:
    n_samples: int = 200
    ci_level: float = 0.95
    random_seed: int | None = None
    noise: NoiseModelConfig = NoiseModelConfig()


@dataclass(frozen=True)
class AnalysisConfig:
    bayesian: BayesianConfig
    uncertainty: UncertaintyConfig


def add_analysis_flags(parser: ArgumentParser) -> None:
    """Expose priors and noise assumptions via CLI flags."""

    parser.add_argument("--bayes-method", choices=["bic", "waic", "loo_proxy"], default="bic")
    parser.add_argument("--prior-pfo", type=float, default=1 / 3, help="Prior probability for PFO")
    parser.add_argument("--prior-pso", type=float, default=1 / 3, help="Prior probability for PSO")
    parser.add_argument("--prior-lh", type=float, default=1 / 3, help="Prior probability for L-H")

    parser.add_argument("--mc-samples", type=int, default=200, help="Monte Carlo synthetic datasets")
    parser.add_argument("--ci-level", type=float, default=0.95, help="Confidence/credible interval level")
    parser.add_argument("--noise-kind", choices=["gaussian", "relative"], default="gaussian")
    parser.add_argument("--noise-sigma", type=float, default=0.05)
    parser.add_argument("--noise-floor", type=float, default=1e-12)
    parser.add_argument("--random-seed", type=int, default=None)


def analysis_config_from_args(args: Namespace) -> AnalysisConfig:
    bayesian = BayesianConfig(
        method=args.bayes_method,
        pfo_prior=args.prior_pfo,
        pso_prior=args.prior_pso,
        lh_prior=args.prior_lh,
    )
    uncertainty = UncertaintyConfig(
        n_samples=args.mc_samples,
        ci_level=args.ci_level,
        random_seed=args.random_seed,
        noise=NoiseModelConfig(
            kind=args.noise_kind,
            sigma=args.noise_sigma,
            floor=args.noise_floor,
        ),
    )
    return AnalysisConfig(bayesian=bayesian, uncertainty=uncertainty)
