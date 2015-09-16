"""
Test functions in stats module.
"""
from __future__ import print_function, division
import numpy as np
import unittest
import stats
import scipy.stats


class TruismsTestCase(unittest.TestCase):

    def test_true_is_false(self):
        self.assertFalse(True)


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

    def test_pearsonr_partial(self):

        x = np.random.normal(0, 1, 200)
        z = x + np.random.normal(0, .2, x.shape)
        y = 3 * z + np.random.normal(0, .1, x.shape)

        rho, p, lb, ub = stats.pearsonr_partial_with_confidence(x, y, [z])

        # find best fit line
        slope, icpt, _, _, _ = scipy.stats.linregress(z, y)
        y_prime = y - (slope * z + icpt)

        rho_correct, p_correct, lb_correct, ub_correct = stats.pearsonr_with_confidence(x, y_prime)

        self.assertAlmostEqual(rho, rho_correct)
        self.assertAlmostEqual(p, p_correct)
        self.assertAlmostEqual(lb, lb_correct)
        self.assertAlmostEqual(ub, ub_correct)

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

    def test_nansem_gives_same_as_scipy_stats_sem(self):

        # 1d array with no nans
        x = np.random.normal(0, 1, 1000)
        self.assertAlmostEqual(scipy.stats.sem(x), stats.nansem(x))

        # 1d array with nans
        x = np.concatenate([x, np.nan * np.ones(100,)])
        self.assertAlmostEqual(scipy.stats.sem(x[:1000]), stats.nansem(x))

        # 2d array with no nans
        x = np.random.normal(0, 2, (50, 50))
        self.assertAlmostEqual(scipy.stats.sem(x, axis=None), stats.nansem(x, axis=None))
        np.testing.assert_array_almost_equal(scipy.stats.sem(x, axis=0), stats.nansem(x, axis=0))
        np.testing.assert_array_almost_equal(scipy.stats.sem(x, axis=1), stats.nansem(x, axis=1))

        # 2d array with nans
        y = x.copy()
        y[-5:, :] = np.nan
        self.assertAlmostEqual(scipy.stats.sem(y[:-5, :], axis=None), stats.nansem(y, axis=None))
        np.testing.assert_array_almost_equal(scipy.stats.sem(y[:-5, :], axis=0),
                                             stats.nansem(y, axis=0))

        y = x.copy()
        y[:, -5:] = np.nan
        self.assertAlmostEqual(scipy.stats.sem(y[:, :-5], axis=None), stats.nansem(y, axis=None))
        np.testing.assert_array_almost_equal(scipy.stats.sem(y[:, :-5], axis=1),
                                             stats.nansem(y, axis=1))


if __name__ == '__main__':
    unittest.main()