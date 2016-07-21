"""
Unit tests for calculating crossings features.
"""
from __future__ import print_function, division
import numpy as np
import unittest
import crossing_features
import time_series


class Crossing(object):
    """
    Mock class for crossings.
    """
    def __init__(self):
        pass


class FeatureFunctionsTestCase(unittest.TestCase):

    def setUp(self):
        print("In test '{}'...".format(self._testMethodName))

        # make odor time-series with three triangular peaks
        odors = np.concatenate([np.zeros((30,), dtype=float),  # total length 30
                                np.arange(1, 21, dtype=float),  # total length 50
                                np.arange(19, -1, -1, dtype=float),  # total length 70
                                np.zeros((50,), dtype=float),  # total length 120
                                np.arange(1, 31, dtype=float),  # total length 150
                                np.arange(29, -1, -1, dtype=float),  # total length 180
                                np.zeros((100,), dtype=float),  # total length 280
                                np.arange(1, 26, dtype=float),  # total length 305
                                np.arange(24, -1, -1, dtype=float),  # total length 330
                                np.zeros((30,), dtype=float),  # total length 360
                                ])

        th = 9.5
        time_vec = np.arange(len(odors) + 2) + 300
        crossing_matrix, peaks = time_series.segment_by_threshold(odors, th, time_vec)

        crossings = []

        for ctr, (row, peak) in enumerate(zip(crossing_matrix, peaks)):
            crossing = Crossing()
            crossing.start_timepoint_id = row[0]
            crossing.entry_timepoint_id = row[1]
            crossing.peak_timepoint_id = row[2]
            crossing.exit_timepoint_id = row[3] - 1
            crossing.end_timepoint_id = row[4] - 1
            crossing.crossing_number = ctr + 1
            crossing.max_odor = peak
            crossings.append(crossing)

        self.odors = odors
        self.traj_start_timepoint_id = time_vec[0]
        self.th = th
        self.crossings = crossings

    def test_max_odor_calculation(self):
        max_odors_correct = [20, 30, 25]
        for crossing, max_odor_correct in zip(self.crossings, max_odors_correct):
            max_odor = crossing_features.max_odor(crossing=crossing,
                                                  odors=self.odors,
                                                  traj_start=self.traj_start_timepoint_id)
            self.assertAlmostEqual(max_odor, max_odor_correct)
            self.assertAlmostEqual(max_odor, crossing.max_odor)

    def test_mean_odor_calculation(self):
        mean_odors_correct = [14.761904761904763,
                              19.756097560975611,
                              17.258064516129032]

        for crossing, mean_odor_correct in zip(self.crossings, mean_odors_correct):
            mean_odor = crossing_features.mean_odor(crossing=crossing,
                                                    odors=self.odors,
                                                    traj_start=self.traj_start_timepoint_id)
            self.assertAlmostEqual(mean_odor, mean_odor_correct)


if __name__ == '__main__':
    unittest.main()