"""
Unit tests for clean_trajectories.py.
"""
from __future__ import print_function, division
import unittest
import numpy as np
from numpy import concatenate as cc
from scripts import clean_trajectories

CLEANING_PARAMS = {'speed_threshold': 0.03,
                   'dist_from_wall_threshold': 0.01,
                   'min_pause_length': 10,
                   'min_trajectory_length': 50}


class Trajectory(object):
    """
    Mock trajectory class.
    """

    def __init__(self, start_timepoint_id, speeds, dists):
        self.start_timepoint_id = start_timepoint_id
        self.end_timepoint_id = start_timepoint_id + len(speeds) - 1
        self.speeds = speeds
        self.dists = dists

    def velocities_a(self, session):
        return self.speeds

    def distances_from_wall(self, session):
        return self.dists


class TruismsTestCase(unittest.TestCase):

    def test_true_is_true(self):
        self.assertTrue(True)

    def test_false_is_true(self):
        self.assertTrue(False)


class TrajectoryCleaningTestCase(unittest.TestCase):

    def test_good_trajectory_is_left_untouched(self):

        speeds = np.random.uniform(0, 1, 500) + 3
        dists = np.random.uniform(0, 5, 500) + 9
        start_timepoint_id = 1790
        traj = Trajectory(start_timepoint_id, speeds, dists)

        clean_portions = clean_trajectories.clean_traj(traj, CLEANING_PARAMS)

        self.assertEqual(len(clean_portions), 1)
        self.assertEqual(clean_portions[0, 0], traj.start_timepoint_id)
        self.assertEqual(clean_portions[0, 1], traj.end_timepoint_id)

    def test_trajectory_with_speed_pause_but_not_distance_pause_is_left_untouched(self):

        speeds = cc([np.random.uniform(0, 1, 500) + 3,
                     np.random.uniform(0, 0.01, 20),
                     np.random.uniform(0, 1, 500) + 3])
        dists = cc([np.random.uniform(0, 1, 500) + 9,
                    np.random.uniform(0, 0.003, 20) + 4,
                    np.random.uniform(0, 1, 500) + 9])
        start_timepoint_id = 10883

        traj = Trajectory(start_timepoint_id, speeds, dists)

        clean_portions = clean_trajectories.clean_traj(traj, CLEANING_PARAMS)

        self.assertEqual(len(clean_portions), 1)
        self.assertEqual(clean_portions[0, 0], traj.start_timepoint_id)
        self.assertEqual(clean_portions[0, 1], traj.end_timepoint_id)

    def test_good_trajectory_but_too_short(self):

        speeds = np.random.uniform(0, 1, 10) + 3
        dists = np.random.uniform(0, 5, 10) + 9
        start_timepoint_id = 4000
        traj = Trajectory(start_timepoint_id, speeds, dists)

        clean_portions = clean_trajectories.clean_traj(traj, CLEANING_PARAMS)

        self.assertEqual(len(clean_portions), 0)

    def test_traj_starting_with_pause_gets_pause_removed(self):

        speeds = cc([np.random.uniform(0, 0.01, 20), np.random.uniform(0, 1, 500) + 3])
        dists = cc([np.random.uniform(0, 0.003, 20), np.random.uniform(0, 1, 500) + 9])
        start_timepoint_id = 2001
        traj = Trajectory(start_timepoint_id, speeds, dists)

        clean_portions = clean_trajectories.clean_traj(traj, CLEANING_PARAMS)

        self.assertEqual(len(clean_portions), 1)
        self.assertEqual(clean_portions[0, 0], start_timepoint_id + 20)
        self.assertEqual(clean_portions[0, 1], traj.end_timepoint_id)

    def test_traj_starting_with_short_pause_is_untouched(self):

        speeds = cc([np.random.uniform(0, 0.01, 5), np.random.uniform(0, 1, 500) + 3])
        dists = cc([np.random.uniform(0, 0.003, 5), np.random.uniform(0, 1, 500) + 9])
        start_timepoint_id = 20010
        traj = Trajectory(start_timepoint_id, speeds, dists)

        clean_portions = clean_trajectories.clean_traj(traj, CLEANING_PARAMS)

        self.assertEqual(len(clean_portions), 1)
        self.assertEqual(clean_portions[0, 0], start_timepoint_id)
        self.assertEqual(clean_portions[0, 1], traj.end_timepoint_id)

    def test_traj_ends_with_long_pause_has_pause_removed(self):

        speeds = cc([np.random.uniform(0, 1, 500) + 3, np.random.uniform(0, 0.01, 20)])
        dists = cc([np.random.uniform(0, 1, 500) + 9, np.random.uniform(0, 0.003, 20)])
        start_timepoint_id = 1008
        traj = Trajectory(start_timepoint_id, speeds, dists)

        clean_portions = clean_trajectories.clean_traj(traj, CLEANING_PARAMS)

        self.assertEqual(len(clean_portions), 1)
        self.assertEqual(clean_portions[0, 0], start_timepoint_id)
        self.assertEqual(clean_portions[0, 1], traj.end_timepoint_id - 20)

    def test_traj_ends_with_short_pause_is_left_untouched(self):

        speeds = cc([np.random.uniform(0, 1, 500) + 3, np.random.uniform(0, 0.01, 5)])
        dists = cc([np.random.uniform(0, 1, 500) + 9, np.random.uniform(0, 0.003, 5)])
        start_timepoint_id = 10081
        traj = Trajectory(start_timepoint_id, speeds, dists)

        clean_portions = clean_trajectories.clean_traj(traj, CLEANING_PARAMS)

        self.assertEqual(len(clean_portions), 1)
        self.assertEqual(clean_portions[0, 0], start_timepoint_id)
        self.assertEqual(clean_portions[0, 1], traj.end_timepoint_id)

    def test_traj_with_pause_in_middle_has_pause_removed(self):

        speeds = cc([np.random.uniform(0, 1, 500) + 3,
                     np.random.uniform(0, 0.01, 20),
                     np.random.uniform(0, 1, 500) + 3])

        dists = cc([np.random.uniform(0, 1, 500) + 9,
                    np.random.uniform(0, 0.003, 20),
                    np.random.uniform(0, 1, 500) + 9])
        start_timepoint_id = 10883

        traj = Trajectory(start_timepoint_id, speeds, dists)

        clean_portions = clean_trajectories.clean_traj(traj, CLEANING_PARAMS)

        self.assertEqual(len(clean_portions), 2)
        self.assertEqual(clean_portions[0, 0], start_timepoint_id)
        self.assertEqual(clean_portions[0, 1], start_timepoint_id + 500 - 1)
        self.assertEqual(clean_portions[1, 0], start_timepoint_id + 520)
        self.assertEqual(clean_portions[1, 1], traj.end_timepoint_id)

    def test_traj_with_short_pause_in_middle_is_left_untouched(self):

        speeds = cc([np.random.uniform(0, 1, 500) + 3,
                     np.random.uniform(0, 0.01, 5),
                     np.random.uniform(0, 1, 500) + 3])

        dists = cc([np.random.uniform(0, 1, 500) + 9,
                    np.random.uniform(0, 0.003, 5),
                    np.random.uniform(0, 1, 500) + 9])

        start_timepoint_id = 108843

        traj = Trajectory(start_timepoint_id, speeds, dists)

        clean_portions = clean_trajectories.clean_traj(traj, CLEANING_PARAMS)

        self.assertEqual(len(clean_portions), 1)
        self.assertEqual(clean_portions[0, 0], start_timepoint_id)
        self.assertEqual(clean_portions[0, 1], traj.end_timepoint_id)

    def test_traj_with_three_pauses_is_cleaned_correctly(self):

        speeds = cc([np.random.uniform(0, 1, 100) + 3,
                     np.random.uniform(0, 0.01, 20),  # long pause
                     np.random.uniform(0, 1, 80) + 3,
                     np.random.uniform(0, 0.01, 5),  # short pause
                     np.random.uniform(0, 1, 95) + 3,
                     np.random.uniform(0, 0.01, 20),  # long pause
                     np.random.uniform(0, 1, 80) + 3])

        dists = cc([np.random.uniform(0, 1, 100) + 3,
                     np.random.uniform(0, 0.01, 20),  # long pause
                     np.random.uniform(0, 1, 80) + 3,
                     np.random.uniform(0, 0.01, 5),  # short pause
                     np.random.uniform(0, 1, 95) + 3,
                     np.random.uniform(0, 0.01, 20),  # long pause
                     np.random.uniform(0, 1, 80) + 3])

        start_timepoint_id = 98

        traj = Trajectory(start_timepoint_id, speeds, dists)

        clean_portions = clean_trajectories.clean_traj(traj, CLEANING_PARAMS)

        self.assertEqual(len(clean_portions), 3)
        self.assertEqual(clean_portions[0, 0], start_timepoint_id)
        self.assertEqual(clean_portions[0, 1], start_timepoint_id + 100 - 1)
        self.assertEqual(clean_portions[1, 0], start_timepoint_id + 120)
        self.assertEqual(clean_portions[1, 1], start_timepoint_id + 300 - 1)
        self.assertEqual(clean_portions[2, 0], start_timepoint_id + 320)
        self.assertEqual(clean_portions[2, 1], traj.end_timepoint_id)

    def test_traj_with_middle_and_end_pause_is_cleaned_correctly(self):

        speeds = cc([np.random.uniform(0, 1, 100) + 3,
                     np.random.uniform(0, 0.01, 20),  # long pause
                     np.random.uniform(0, 1, 80) + 3,
                     np.random.uniform(0, 0.01, 20)])  # long pause

        dists = cc([np.random.uniform(0, 1, 100) + 3,
                     np.random.uniform(0, 0.01, 20),  # long pause
                     np.random.uniform(0, 1, 80) + 3,
                     np.random.uniform(0, 0.01, 20)])  # long pause

        start_timepoint_id = 9809

        traj = Trajectory(start_timepoint_id, speeds, dists)

        clean_portions = clean_trajectories.clean_traj(traj, CLEANING_PARAMS)

        self.assertEqual(len(clean_portions), 2)
        self.assertEqual(clean_portions[0, 0], start_timepoint_id)
        self.assertEqual(clean_portions[0, 1], start_timepoint_id + 100 - 1)
        self.assertEqual(clean_portions[1, 0], start_timepoint_id + 120)
        self.assertEqual(clean_portions[1, 1], traj.end_timepoint_id - 20)

    def test_two_long_pauses_in_middle_with_too_short_traj_between_is_cleaned_correctly(self):

        speeds = cc([np.random.uniform(0, 1, 100) + 3,
                     np.random.uniform(0, 0.01, 20),  # long pause
                     np.random.uniform(0, 1, 30) + 3,  # short trajectory
                     np.random.uniform(0, 0.01, 20),  # long pause
                     np.random.uniform(0, 1, 80) + 3])

        dists = cc([np.random.uniform(0, 1, 100) + 3,
                     np.random.uniform(0, 0.01, 20),  # long pause
                     np.random.uniform(0, 1, 30) + 3,  # short trajectory
                     np.random.uniform(0, 0.01, 20),  # long pause
                     np.random.uniform(0, 1, 80) + 3])

        start_timepoint_id = 1003

        traj = Trajectory(start_timepoint_id, speeds, dists)

        clean_portions = clean_trajectories.clean_traj(traj, CLEANING_PARAMS)

        self.assertEqual(len(clean_portions), 2)
        self.assertEqual(clean_portions[0, 0], start_timepoint_id)
        self.assertEqual(clean_portions[0, 1], start_timepoint_id + 100 -1)
        self.assertEqual(clean_portions[1, 0], start_timepoint_id + 170)
        self.assertEqual(clean_portions[1, 1], traj.end_timepoint_id)


if __name__ == '__main__':
    unittest.main()