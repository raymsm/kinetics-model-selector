from __future__ import annotations

import unittest

from kinetics_model_selector.mathutils import linspace
from kinetics_model_selector.models import langmuir_hinshelwood, pfo, pso


class ParameterRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.t = linspace(0.5, 20.0, 40)

    def assert_close(self, actual: float, expected: float, rel_tol: float = 0.15) -> None:
        self.assertLessEqual(abs(actual - expected), rel_tol * expected)

    def test_pfo_recovery(self) -> None:
        q = pfo.predict(self.t, qe=10.0, k1=0.15)
        fit = pfo.fit(self.t, q)
        self.assert_close(fit["qe"], 10.0, rel_tol=0.12)
        self.assert_close(fit["k1"], 0.15, rel_tol=0.12)

    def test_pso_recovery(self) -> None:
        q = pso.predict(self.t, qe=8.5, k2=0.08)
        fit = pso.fit(self.t, q)
        self.assert_close(fit["qe"], 8.5, rel_tol=0.01)
        self.assert_close(fit["k2"], 0.08, rel_tol=0.01)

    def test_lh_recovery(self) -> None:
        q = langmuir_hinshelwood.predict(self.t, k=3.0, K=0.2)
        fit = langmuir_hinshelwood.fit(self.t, q)
        self.assert_close(fit["k"], 3.0, rel_tol=0.01)
        self.assert_close(fit["K"], 0.2, rel_tol=0.01)


if __name__ == "__main__":
    unittest.main()
