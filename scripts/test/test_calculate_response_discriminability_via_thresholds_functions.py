"""
Tests for the time-averaged response calculation function used in
calculate_response_discriminability_via_thresholds.py.
"""
from __future__ import print_function, division
import numpy as np
import unittest
from scripts.calculate_response_discriminability_via_thresholds \
    import get_time_avg_response_diff_and_bounds


class TimeAvgResponseDiffTestCase(unittest.TestCase):

    def test_examples_are_calculated_correctly(self):

        # make two sets of "crossing responses"
        t = np.arange(100, dtype=float)
        x_below = np.tile(-.5 * t, (100, 1))
        x_above = np.tile(-2 * t, (80, 1))

        x_below += np.random.normal(0, 5, x_below.shape)
        x_above += np.random.normal(0, 5, x_above.shape)

        # randomly insert nans
        x_below[np.random.rand(*x_below.shape) > 0.85] = np.nan
        x_above[np.random.rand(*x_above.shape) > 0.85] = np.nan

        # make sure nans got inserted
        self.assertGreater(np.isnan(x_below).sum(), 0)
        self.assertGreater(np.isnan(x_above).sum(), 0)

        # calculate time-averaged difference
        diff, lb, ub = get_time_avg_response_diff_and_bounds(x_below, x_above)

        self.assertGreater(diff, 0)
        self.assertGreater(diff, lb)
        self.assertLess(diff, ub)

        # do the same thing but make sure offsetting x_mean_below doesn't change the results
        x_below -= 1000

        diff, lb, ub = get_time_avg_response_diff_and_bounds(x_below, x_above)

        self.assertGreater(diff, 0)
        self.assertGreater(diff, lb)
        self.assertLess(diff, ub)

        # make sure we get the opposite if we feed in the reverse pair
        diff, lb, ub = get_time_avg_response_diff_and_bounds(x_above, x_below)

        self.assertLess(diff, 0)
        self.assertGreater(diff, lb)
        self.assertLess(diff, ub)


if __name__ == '__main__':
    unittest.main()