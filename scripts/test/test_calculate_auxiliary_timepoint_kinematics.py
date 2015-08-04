"""
Test that the script "calculate_auxiliary_timepoint_kinematics.py" worked correctly.
"""
from __future__ import print_function, division
import unittest
import numpy as np
from db_api import models
import kinematics
from scripts.calculate_auxiliary_timepoint_kinematics import main, session

N_TRAJS_PER_EXPT_TO_TEST = 10
WALL_BOUNDS = ((-0.3, 1.), (-0.15, 0.15), (-0.15, 0.15))


class MainTestCase(unittest.TestCase):

    def setUp(self):
        print("In test '{}'".format(self._testMethodName))
        self.expts = session.query(models.Experiment)
        self.dt = 0.01

    def test_no_nans(self):
        for expt in self.expts:
            for traj in expt.trajectories:
                self.assertFalse(np.any(np.isnan(traj.positions(session))))
                self.assertFalse(np.any(np.isnan(traj.velocities(session))))
                self.assertFalse(np.any(np.isnan(traj.velocities_a(session))))
                self.assertFalse(np.any(np.isnan(traj.accelerations(session))))
                self.assertFalse(np.any(np.isnan(traj.accelerations_a(session))))
                self.assertFalse(np.any(np.isnan(traj.headings(session))))
                self.assertFalse(np.any(np.isnan(traj.angular_velocities(session))))
                self.assertFalse(np.any(np.isnan(traj.angular_velocities_a(session))))
                self.assertFalse(np.any(np.isnan(traj.angular_accelerations(session))))
                self.assertFalse(np.any(np.isnan(traj.angular_accelerations_a(session))))
                self.assertFalse(np.any(np.isnan(traj.distances_from_wall(session))))

    def test_kinematics_calculated_correctly(self):
        for expt in self.expts:
            for traj in expt.trajectories[:N_TRAJS_PER_EXPT_TO_TEST]:
                positions = traj.positions(session)
                velocities = traj.velocities(session)
                velocities_a = traj.velocities_a(session)
                accelerations = traj.accelerations(session)
                accelerations_a = traj.accelerations_a(session)
                headings = traj.headings(session)
                angular_velocities = traj.angular_velocities(session)
                angular_velocities_a = traj.angular_velocities_a(session)
                angular_accelerations = traj.angular_accelerations(session)
                angular_accelerations_a = traj.angular_accelerations_a(session)
                distances_from_wall = traj.distances_from_wall(session)

                np.testing.assert_array_almost_equal(velocities_a, kinematics.norm(velocities),
                                                     decimal=4)
                np.testing.assert_array_almost_equal(accelerations, kinematics.acceleration(velocities, self.dt),
                                                     decimal=4)
                np.testing.assert_array_almost_equal(accelerations_a, kinematics.norm(accelerations),
                                                     decimal=4)
                np.testing.assert_array_almost_equal(headings, kinematics.heading(velocities),
                                                     decimal=3)
                np.testing.assert_array_almost_equal(angular_velocities, kinematics.angular_velocity(velocities, self.dt),
                                                     decimal=3)
                np.testing.assert_array_almost_equal(angular_velocities_a, kinematics.norm(angular_velocities),
                                                     decimal=3)
                np.testing.assert_array_almost_equal(angular_accelerations,
                                                     kinematics.acceleration(angular_velocities, self.dt),
                                                     decimal=1)
                np.testing.assert_array_almost_equal(angular_accelerations_a, kinematics.norm(angular_accelerations),
                                                     decimal=1)
                np.testing.assert_array_almost_equal(distances_from_wall,
                                                     kinematics.distance_from_wall(positions, WALL_BOUNDS),
                                                     decimal=4)


if __name__ == '__main__':
    print("Running 'main()'...")
    main()
    print("Running tests...")
    unittest.main()