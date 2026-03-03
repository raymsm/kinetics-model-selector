import numpy as np

from selection.bayesian import posterior_model_probabilities
from uncertainty.monte_carlo import NoiseModelConfig, propagate_monte_carlo_uncertainty


class DummyBayesModel:
    def __init__(self, name, k, n, ll):
        self.name = name
        self.n_parameters = k
        self.n_observations = n
        self.log_likelihood = ll


class LinearRefit:
    def __init__(self, name="PFO"):
        self.name = name
        self._params = {"a": 0.0, "b": 0.0}

    def copy(self):
        c = LinearRefit(self.name)
        c._params = self._params.copy()
        return c

    def fit(self, x, y):
        b, a = np.polyfit(x, y, 1)
        self._params = {"a": float(a), "b": float(b)}

    def predict(self, x):
        return self._params["a"] + self._params["b"] * x

    @property
    def parameters(self):
        return self._params


def test_posterior_model_probabilities_normalized():
    models = [
        DummyBayesModel("PFO", 2, 20, -10.0),
        DummyBayesModel("PSO", 3, 20, -10.5),
        DummyBayesModel("L-H", 4, 20, -11.0),
    ]
    scores = posterior_model_probabilities(models, method="bic")
    total = sum(s.posterior_probability for s in scores)
    assert np.isclose(total, 1.0)
    assert scores[0].model_name == "PFO"


def test_monte_carlo_outputs_intervals():
    x = np.linspace(0, 10, 25)
    y = 2.0 + 0.5 * x
    model = LinearRefit("PFO")
    summary = propagate_monte_carlo_uncertainty(
        [model],
        x,
        y,
        n_samples=20,
        noise=NoiseModelConfig(kind="gaussian", sigma=0.1),
        ci_level=0.9,
        random_seed=1,
    )
    assert "PFO" in summary.parameter_intervals
    assert "a" in summary.parameter_intervals["PFO"]
    assert summary.prediction_intervals["PFO"]["median"].shape == x.shape
