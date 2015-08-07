"""
Unit tests for function used to make distribution of kinematic quantities.
"""
from __future__ import print_function, division
import unittest
import numpy as np
from scripts.calculate_kinematic_distributions import make_distribution


class DistributionsTestCase(unittest.TestCase):

    def test_bins_are_approximately_correctly_calculated(self):

        # example unbounded
        v = []
        n_total = 0
        for _ in range(20):
            n = np.random.randint(200, 1000)
            v.append(np.random.normal(1, 2, n))
            n_total += n

        cts, bins = make_distribution(np.concatenate(v), n_bins=20)

        self.assertAlmostEqual(cts.sum(), n_total, delta=10)
        self.assertAlmostEqual(bins[0], -9, delta=0.3)
        self.assertAlmostEqual(bins[-1], 11, delta=0.3)

        # example bounded
        v = []
        n_total = 0
        for _ in range(20):
            n = np.random.randint(200, 1000)
            v.append(np.random.exponential(2, n))
            n_total += n

        cts, bins = make_distribution(np.concatenate(v), n_bins=20, lb=0)

        self.assertAlmostEqual(cts.sum(), n_total, delta=45)
        self.assertAlmostEqual(bins[0], 0, places=5)
        self.assertAlmostEqual(bins[-1], 12, delta=0.3)

        # another example bounded
        v = []
        n_total = 0
        for _ in range(20):
            n = np.random.randint(200, 1000)
            v.append(np.random.exponential(2, n))
            n_total += n

        cts, bins = make_distribution(np.concatenate(v), n_bins=20, lb=0, ub=40)

        self.assertAlmostEqual(cts.sum(), n_total, delta=5)
        self.assertAlmostEqual(bins[0], 0, places=5)
        self.assertAlmostEqual(bins[-1], 40, places=5)


if __name__ == '__main__':
    unittest.main()