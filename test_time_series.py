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

    def test_segment_basic_segments_correctly_with_external_idxs(self):

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

    def test_segment_by_threshold_gives_correct_answer_for_examples(self):

        threshold = 5

        t = np.arange(100, 200)
        t_extended = np.arange(100, 201)

        # signal with two threshold crossings
        x = np.random.uniform(0, 2, t.shape)
        x[10:20] = 10 + np.random.uniform(0, 2, (10,))  # above threshold
        x[60:75] = 20 + np.random.uniform(0, 2, (15,))  # above threshold

        x[15] = 15  # peak 1
        x[70] = 25  # peak 2

        # correct solution
        segments_correct = np.array([[100, 110, 115, 120, 160],
                                     [120, 160, 170, 175, 200]])
        peaks_correct = np.array([15., 25])

        segments, peaks = time_series.segment_by_threshold(x, threshold, t=t_extended)

        np.testing.assert_array_equal(segments, segments_correct)
        np.testing.assert_array_equal(peaks, peaks_correct)

        # signal ending above threshold
        x = np.random.uniform(0, 2, t.shape)
        x[10:20] = 10 + np.random.uniform(0, 2, (10,))  # above threshold
        x[60:] = 20 + np.random.uniform(0, 2, (40,))  # above threshold

        x[15] = 15  # peak 1
        x[70] = 25  # peak 2

        # correct solution
        segments_correct = np.array([[100, 110, 115, 120, 160],
                                     [120, 160, 170, 200, 200]])
        peaks_correct = np.array([15., 25])

        segments, peaks = time_series.segment_by_threshold(x, threshold, t=t_extended)

        np.testing.assert_array_equal(segments, segments_correct)
        np.testing.assert_array_equal(peaks, peaks_correct)

        # signal starting above threshold
        x = np.random.uniform(0, 2, t.shape)
        x[:20] = 10 + np.random.uniform(0, 2, (20,))  # above threshold
        x[60:75] = 20 + np.random.uniform(0, 2, (15,))  # above threshold

        x[15] = 15  # peak 1
        x[70] = 25  # peak 2

        # correct solution
        segments_correct = np.array([[100, 100, 115, 120, 160],
                                     [120, 160, 170, 175, 200]])
        peaks_correct = np.array([15., 25])

        segments, peaks = time_series.segment_by_threshold(x, threshold, t=t_extended)

        np.testing.assert_array_equal(segments, segments_correct)
        np.testing.assert_array_equal(peaks, peaks_correct)

    def test_segment_by_threshold_gives_correct_answer_for_one_or_zero_threshold_crossings(self):

        threshold = 5

        # no crossings
        x = np.random.uniform(0, 1, 100)

        segments, peaks = time_series.segment_by_threshold(x, threshold)

        self.assertEqual(segments.shape, (0, 5))
        self.assertEqual(len(peaks), 0)

        # one crossing
        x[50:60] = 10
        x[55] = 15

        segments_correct = np.array([[0, 50, 55, 60, 100]])
        peaks_correct = np.array([15])

        segments, peaks = time_series.segment_by_threshold(x, threshold)

        np.testing.assert_array_equal(segments, segments_correct)
        np.testing.assert_array_equal(peaks, peaks_correct)


class CrossCovarianceTestCase(unittest.TestCase):

    def setUp(self):
        print("In test '{}'...".format(self._testMethodName))

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

        acov, _, lb, ub = time_series.\
            xcov_multi_with_confidence(xs, xs, lag_backward=20, lag_forward=21, normed=True)
        acov_smooth, _, lb_smooth, ub_smooth = time_series.\
            xcov_multi_with_confidence(xs_smooth, xs_smooth, lag_backward=20, lag_forward=21, normed=True)

        for lag in range(2, 21):
            self.assertLess(acov[20 + lag], acov_smooth[20 + lag])
            self.assertLess(acov[20 + lag], acov_smooth[20 + lag])
            self.assertLess(acov[20 - lag], acov_smooth[20 - lag])
            self.assertLess(acov[20 - lag], acov_smooth[20 - lag])

            self.assertGreater(acov_smooth[20 + lag], lb_smooth[20 + lag])
            self.assertLess(acov_smooth[20 + lag], ub_smooth[20 + lag])
            self.assertGreater(acov_smooth[20 - lag], lb_smooth[20 - lag])
            self.assertLess(acov_smooth[20 - lag], ub_smooth[20 - lag])

        np.testing.assert_array_less(lb[:20], ub[:20])
        self.assertTrue(np.all(lb_smooth[:20] > -1))
        self.assertTrue(np.all(ub_smooth[:20] < 1))

    def test_unnormed_autocovariance_gives_variance_at_0_lag(self):

        xs = [np.random.normal(0, 1, np.random.randint(500, 1000)) for _ in range(100)]
        xs = [ndimage.gaussian_filter1d(x, 5) for x in xs]

        acov, _, lb, ub = time_series.\
            xcov_multi_with_confidence(xs, xs, lag_backward=20, lag_forward=21, normed=False)

        self.assertAlmostEqual(acov[20], np.var(np.concatenate(xs)), places=5)

        for lag in range(2, 21):
            self.assertGreater(acov[20 + lag], lb[20 + lag])
            self.assertLess(acov[20 + lag], ub[20 + lag])
            self.assertGreater(acov[20 - lag], lb[20 - lag])
            self.assertLess(acov[20 - lag], ub[20 - lag])


if __name__ == '__main__':
    unittest.main()