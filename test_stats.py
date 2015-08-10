"""
Test functions in stats module.
"""
from __future__ import print_function, division
import numpy as np
import unittest
import stats


class StatsTestCase(unittest.TestCase):

    def test_pearsonr_with_confidence(self):
        for noise in [0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10]:
            x = np.random.uniform(0, 1, 100)
            y = x + np.random.normal(0, noise, 100)

            rho, p, lb, ub = stats.pearsonr_with_confidence(x, y)

            self.assertGreater(ub, rho)
            self.assertLess(lb, rho)

            self.assertLess(ub, 1)
            self.assertGreater(lb, -1)

            if p < 0.05:
                self.assertGreater(lb, 0)
            else:
                self.assertLess(lb, 0)

        for noise in [0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10]:
            x = np.random.uniform(0, 1, 100)
            y = x**4 + np.random.normal(0, noise, 100)

            rho, p, lb, ub = stats.pearsonr_with_confidence(x, y)

            self.assertGreater(ub, rho)
            self.assertLess(lb, rho)

            self.assertLess(ub, 1)
            self.assertGreater(lb, -1)

            if p < 0.05:
                self.assertGreater(lb, 0)
            else:
                self.assertLess(lb, 0)

    def test_cov_with_confidence(self):
        for noise in [0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10]:

            x = np.random.normal(0, 1, 1000)
            y = 3 * x + np.random.normal(0, noise, x.shape)

            cov, pv, lb, ub = stats.cov_with_confidence(x, y, confidence=0.95)

            self.assertGreater(cov, 1)
            self.assertGreater(cov, lb)
            self.assertLess(cov, ub)

        x = np.random.uniform(0, 10, 1000)
        y = 3 * x + np.random.normal(0, 0.01, x.shape)
        cov, pv, lb, ub = stats.cov_with_confidence(x, y, confidence=0.95)

        self.assertGreater(cov, 1)


if __name__ == '__main__':
    unittest.main()