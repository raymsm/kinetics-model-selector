from __future__ import annotations

import unittest

from kinetics_model_selector.mathutils import linspace
from kinetics_model_selector.models import pso
from kinetics_model_selector.uncertainty import monte_carlo_parameter_samples


class UncertaintyTests(unittest.TestCase):
    def test_monte_carlo_shape_and_intervals(self) -> None:
        t = linspace(0.5, 20.0, 40)
        q = pso.predict(t, qe=8.5, k2=0.08)

        samples, intervals = monte_carlo_parameter_samples(
            t,
            q,
            pso.fit,
            ["qe", "k2"],
            n_iter=50,
            noise_std=0.05,
            random_seed=101,
        )

        self.assertEqual(len(samples), 50)
        self.assertEqual(len(samples[0]), 2)
        self.assertEqual(len(intervals), 2)
        for lower, upper in intervals:
            self.assertLess(lower, upper)
            self.assertTrue(lower > 0)


if __name__ == "__main__":
    unittest.main()
