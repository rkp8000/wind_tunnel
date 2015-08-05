"""
Unit tests for time-series module.
"""
from __future__ import print_function, division
import unittest
import numpy as np
import time_series


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

if __name__ == '__main__':
    unittest.main()