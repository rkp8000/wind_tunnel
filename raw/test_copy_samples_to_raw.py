"""
Test to make sure that samples were correctly copied from raw database.
"""
from __future__ import print_function, division
import unittest
import numpy as np
from connect import session, session_sample
import models

N_TIMEPOINTS_TO_COMPARE = 20


class TruismsTestCase(unittest.TestCase):

    def setUp(self):
        print("In test '{}'...".format(self._testMethodName))

    def test_false_is_true(self):
        self.assertTrue(False)

    def test_true_is_true(self):
        self.assertTrue(True)


class CorrectTrajectoriesTestCase(unittest.TestCase):

    def setUp(self):
        print("In test '{}'...".format(self._testMethodName))

    def test_sample_trajs_have_same_fields_and_timepoints_as_trajs_in_original_db(self):

        # loop over experiments
        for experiment in session_sample.query(models.Experiment):

            print("Testing trajectories from experiment '{}'...".format(experiment.id))

            for traj_sample in experiment.trajectories:
                # make sure is raw and not clean
                self.assertTrue(traj_sample.raw)
                self.assertFalse(traj_sample.clean)
                # get partner trajectory in original database
                traj_original = session.query(models.Trajectory).get(traj_sample.id)

                # make sure odor states match
                self.assertEqual(traj_sample.odor_state, traj_original.odor_state)

                # make sure they have the same number of timepoints
                self.assertEqual(traj_sample.end_timepoint_id - traj_sample.start_timepoint_id,
                                 traj_original.end_timepoint_id - traj_original.start_timepoint_id)

                # make sure a few of the timepoints are identical
                traj_len = traj_sample.end_timepoint_id - traj_sample.start_timepoint_id + 1
                rand_idxs = np.random.randint(0, traj_len, N_TIMEPOINTS_TO_COMPARE)

                positions_sample = traj_sample.positions(session_sample)
                positions_original = traj_original.positions(session)
                velocities_sample = traj_sample.velocities(session_sample)
                velocities_original = traj_original.velocities(session)
                odors_sample = traj_sample.odors(session_sample)
                odors_original = traj_original.odors(session)

                for idx in rand_idxs:
                    for dim in range(3):
                        self.assertAlmostEqual(positions_sample[idx][dim], positions_original[idx][dim])
                        self.assertAlmostEqual(velocities_sample[idx][dim], velocities_original[idx][dim])
                    self.assertAlmostEqual(odors_sample[idx], odors_original[idx])


if __name__ == '__main__':
    unittest.main()