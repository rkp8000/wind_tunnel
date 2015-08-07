"""
Unit tests for time-series module.
"""
from __future__ import print_function, division
import unittest
import numpy as np
import time_series
from scipy import ndimage


class SegmentingTestCase(unittest.TestCase):

    def test_segment_basic_segments_correctly(self):
        """
        A few example time-serieses to be segmented.
        """

        ts = np.array([False, False, True, True, False, False, True, False])
        starts_correct, ends_correct = (np.array([2, 6]), np.array([4, 7]))
        starts, ends = time_series.segment_basic(ts)

        np.testing.assert_array_equal(starts, starts_correct)
        np.testing.assert_array_equal(ends, ends_correct)

        ts = np.array([False, False, True, True, False, False, True, True])
        starts_correct, ends_correct = (np.array([2, 6]), np.array([4, 8]))
        starts, ends = time_series.segment_basic(ts)

        np.testing.assert_array_equal(starts, starts_correct)
        np.testing.assert_array_equal(ends, ends_correct)

        ts = np.array([True, False, True, True, False, False, True, True])
        starts_correct, ends_correct = (np.array([0, 2, 6]), np.array([1, 4, 8]))
        starts, ends = time_series.segment_basic(ts)

        np.testing.assert_array_equal(starts, starts_correct)
        np.testing.assert_array_equal(ends, ends_correct)

    def test_segment_basic_segments_correctly_with_external_indxs(self):

        t = np.arange(100, 200)

        ts = np.array([False, False, True, True, False, False, True, False])
        starts_correct, ends_correct = (np.array([2, 6]) + 100, np.array([4, 7]) + 100)
        starts, ends = time_series.segment_basic(ts, t)

        np.testing.assert_array_equal(starts, starts_correct)
        np.testing.assert_array_equal(ends, ends_correct)

        ts = np.array([False, False, True, True, False, False, True, True])
        starts_correct, ends_correct = (np.array([2, 6]) + 100, np.array([4, 8]) + 100)
        starts, ends = time_series.segment_basic(ts, t)

        np.testing.assert_array_equal(starts, starts_correct)
        np.testing.assert_array_equal(ends, ends_correct)

        ts = np.array([True, False, True, True, False, False, True, True])
        starts_correct, ends_correct = (np.array([0, 2, 6]) + 100, np.array([1, 4, 8]) + 100)
        starts, ends = time_series.segment_basic(ts, t)

        np.testing.assert_array_equal(starts, starts_correct)
        np.testing.assert_array_equal(ends, ends_correct)


class CrossCovarianceTestCase(unittest.TestCase):

    def test_autocovariance_is_delta_function_for_white_noise(self):

        xs = [np.random.normal(0, 3, np.random.randint(500, 1000)) for _ in range(100)]
        acov, pv, conf_lb, conf_ub = \
            time_series.xcov_multi_with_confidence(xs, xs, lag_backward=20, lag_forward=21, normed=True)

        self.assertEqual(len(acov), 41)
        self.assertAlmostEqual(acov[:20].mean(), 0, places=2)
        self.assertAlmostEqual(acov[21:].mean(), 0, places=2)
        self.assertAlmostEqual(acov[20], 1, places=5)

        np.testing.assert_array_almost_equal(acov[:20], acov[21:][::-1])
        np.testing.assert_array_almost_equal(pv[:20], pv[21:][::-1])
        np.testing.assert_array_almost_equal(conf_lb[:20], conf_lb[21:][::-1])
        np.testing.assert_array_almost_equal(conf_ub[:20], conf_ub[21:][::-1])

    def test_autocovariance_is_extended_for_smoother_signals_and_has_correct_confidences(self):

        xs = [np.random.normal(0, 1, np.random.randint(500, 1000)) for _ in range(100)]
        xs_smooth = [ndimage.gaussian_filter1d(x, 5) for x in xs]

        acov, _, _, _ = time_series.\
            xcov_multi_with_confidence(xs, xs, lag_backward=20, lag_forward=21, normed=True)
        acov_smooth, _, lb, ub = time_series.\
            xcov_multi_with_confidence(xs_smooth, xs_smooth, lag_backward=20, lag_forward=21, normed=True)

        for lag in range(2, 11):
            self.assertLess(acov[20 + lag], acov_smooth[20 + lag])
            self.assertLess(acov[20 + lag], acov_smooth[20 + lag])
            self.assertLess(acov[20 - lag], acov_smooth[20 - lag])
            self.assertLess(acov[20 - lag], acov_smooth[20 - lag])

        np.testing.assert_array_less(lb[:20], ub[:20])
        self.assertTrue(np.all(lb[:20] > -1))
        self.assertTrue(np.all(ub[:20] < 1))

    def test_unnormed_autocovariance_gives_variance_at_0_lag(self):

        xs = [np.random.normal(0, 1, np.random.randint(500, 1000)) for _ in range(100)]
        xs = [ndimage.gaussian_filter1d(x, 5) for x in xs]

        acov, _, _, _ = time_series.\
            xcov_multi_with_confidence(xs, xs, lag_backward=20, lag_forward=21, normed=False)

        self.assertAlmostEqual(acov[20], np.var(np.concatenate(xs)), places=5)


if __name__ == '__main__':
    unittest.main()