from __future__ import annotations

import random
import unittest

from kinetics_model_selector.mathutils import linspace
from kinetics_model_selector.models import pfo
from kinetics_model_selector.selection import select_best_model


class SelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.t = linspace(0.5, 20.0, 40)

    def test_best_fit_clean_identification(self) -> None:
        q = pfo.predict(self.t, qe=10.0, k1=0.15)
        best_name, _ = select_best_model(self.t, q)
        self.assertEqual(best_name, "pfo")

    def test_best_fit_noisy_identification(self) -> None:
        q = pfo.predict(self.t, qe=10.0, k1=0.15)
        rng = random.Random(11)
        noisy_q = [qi + rng.gauss(0.0, 0.03 * max(q)) for qi in q]
        best_name, _ = select_best_model(self.t, noisy_q)
        self.assertEqual(best_name, "pfo")


if __name__ == "__main__":
    unittest.main()
