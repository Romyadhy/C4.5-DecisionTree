import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.metrics import calculate_entropy, calculate_gain_raio


class TestMetrics(unittest.TestCase):
    def test_entropy_pure(self):
        y = np.array([0, 0, 0, 0])
        self.assertEqual(calculate_entropy(y), 0)

    def test_entropy_mixed(self):
        y = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(calculate_entropy(y), 1.0)

    def test_gain_ratio_ideal(self):
        parent = np.array([0, 0, 1, 1])
        left = np.array([0, 0])  # Pure
        right = np.array([1, 1])  # Pure

        gr = calculate_gain_raio(parent, [left, right])
        self.assertGreater(gr, 0)  # Gain ratio must be positive


if __name__ == "__main__":
    unittest.main()
