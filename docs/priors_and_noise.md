# Priors and measurement-noise assumptions

This project supports Bayesian model probability approximation for candidate
kinetic models (PFO/PSO/L-H) and Monte Carlo uncertainty propagation.

## Model priors

CLI flags:

- `--prior-pfo`
- `--prior-pso`
- `--prior-lh`

These priors are interpreted as prior model probabilities and combined with a
chosen evidence proxy (`--bayes-method`: `bic`, `waic`, or `loo_proxy`).

## Measurement-noise assumptions

CLI flags:

- `--noise-kind {gaussian,relative}`
- `--noise-sigma`
- `--noise-floor`

### Gaussian noise

`y_syn = y_hat + Normal(0, sigma)`

### Relative noise

`y_syn = y_hat + Normal(0, max(|y_hat| * sigma, floor))`

Used by Monte Carlo propagation (`--mc-samples`) to produce parameter and
prediction intervals at the requested level (`--ci-level`).
