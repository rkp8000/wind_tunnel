"""
Unit tests for kinematics calculations.
"""
from __future__ import print_function, division
import unittest
import numpy as np
import kinematics


class TruismsTestCase(unittest.TestCase):

    def setUp(self):
        print("In test '{}'...".format(self._testMethodName))

    def test_false_is_true(self):
        # this should fail
        self.assertTrue(False)

    def test_true_is_true(self):
        self.assertTrue(True)


class AccelerationTestCase(unittest.TestCase):

    def setUp(self):
        print("In test '{}'...".format(self._testMethodName))

    def test_example_velocities_are_correctly_differentiated(self):

        v = np.array([[1., 1, 1], [2, 2, 2], [3, 3, 3], [5, 5, 5], [7, 7, 7], [10, 10, 10]])
        dt = 0.1
        a_correct = np.array([[1., 1, 1], [1, 1, 1], [1.5, 1.5, 1.5], [2, 2, 2], [2.5, 2.5, 2.5], [3, 3, 3]]) / dt

        np.testing.assert_array_almost_equal(kinematics.acceleration(v, dt), a_correct)


class HeadingTestCase(unittest.TestCase):

    def setUp(self):
        print("In test '{}'...".format(self._testMethodName))

    def test_example_headings_correctly_calculated(self):

        # flight along the axes
        v = np.array([[-1., 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
        h_correct = np.array([[0, 0, 0],
                              [180, 180, 180],
                              [90, 0, 90],
                              [90, 0, 90],
                              [0, 90, 90],
                              [0, 90, 90]])
        self.assertFalse(np.any(np.isnan(kinematics.heading(v))))
        np.testing.assert_array_almost_equal(kinematics.heading(v), h_correct)

        v = np.array([[-1, 1, 0], [-1, -1, 0], [-1, 0, 1], [-1, 0, -1], [-1, 1, 1], [-1, -1, -1]])
        h_correct = np.array([[45, 0, 45],
                              [45, 0, 45],
                              [0, 45, 45],
                              [0, 45, 45],
                              [45, 45, 54.735610317245346],
                              [45, 45, 54.735610317245346]])
        self.assertFalse(np.any(np.isnan(kinematics.heading(v))))
        np.testing.assert_array_almost_equal(kinematics.heading(v), h_correct)

        v = np.array([[1, -1, 0], [1, 0, 1], [1, 1, 1]])
        h_correct = np.array([[135, 180, 135],
                              [180, 135, 135],
                              [135, 135, 125.26438968275465]])
        self.assertFalse(np.any(np.isnan(kinematics.heading(v))))
        np.testing.assert_array_almost_equal(kinematics.heading(v), h_correct)

    def test_all_headings_between_zero_and_180(self):

        # make random velocity vector
        v = np.random.normal(0, 10, (1000, 3))
        h = kinematics.heading(v)
        self.assertFalse(np.any(np.isnan(h)))
        np.testing.assert_array_less(-.00001 * np.ones(h.shape), h)
        np.testing.assert_array_less(h, 180.00001 * np.ones(h.shape))


class AngularVelocityTestCase(unittest.TestCase):

    def setUp(self):
        print("In test '{}'...".format(self._testMethodName))

    def test_angular_velocity_higher_during_turns(self):

        dt = 0.1

        # turn around in xy plane
        v = np.array([[-1., 0, 0], [-1, 0, 0], [-1, 0, 0],
                      [-1, 1, 0], [0, 1, 0], [1, 1, 0],
                      [1, 0, 0], [1, 0, 0], [1, 0, 0]])
        av = kinematics.angular_velocity(v, dt)

        self.assertFalse(np.any(np.isnan(av)))
        # make sure angular velocity in x-direction is zero
        np.testing.assert_array_almost_equal(av[:, 0], np.zeros(av[:, 0].shape))
        # make sure angular velocity in y-direction is zero
        np.testing.assert_array_almost_equal(av[:, 1], np.zeros(av[:, 1].shape))
        # make sure angular velocity magnitude in z-direction is higher in middle
        self.assertGreater(np.abs(av[4, 2]), av[0, 2])
        self.assertGreater(np.abs(av[4, 2]), av[8, 2])
        # make sure angular velocity in middle is less than zero (right-hand rule)
        self.assertLess(av[4, 2], 0)

        # turn around in xy plane the other way
        v = np.array([[-1., 0, 0], [-1, 0, 0], [-1, 0, 0],
                      [-1, -1, 0], [0, -1, 0], [1, -1, 0],
                      [1, 0, 0], [1, 0, 0], [1, 0, 0]])
        av = kinematics.angular_velocity(v, dt)

        self.assertFalse(np.any(np.isnan(av)))
        # make sure angular velocity in x-direction is zero
        np.testing.assert_array_almost_equal(av[:, 0], np.zeros(av[:, 0].shape))
        # make sure angular velocity in y-direction is zero
        np.testing.assert_array_almost_equal(av[:, 1], np.zeros(av[:, 1].shape))
        # make sure angular velocity in z-direction is higher in middle
        self.assertGreater(np.abs(av[4, 2]), av[0, 2])
        self.assertGreater(np.abs(av[4, 2]), av[8, 2])
        # make sure angular velocity in middle is greater than zero (right-hand rule)
        self.assertGreater(av[4, 2], 0)

        # turn around in xz plane
        v = np.array([[-1., 0, 0], [-1, 0, 0], [-1, 0, 0],
                      [-1, 0, 1], [0, 0, 1], [1, 0, 1],
                      [1, 0, 0], [1, 0, 0], [1, 0, 0]])
        av = kinematics.angular_velocity(v, dt)

        self.assertFalse(np.any(np.isnan(av)))
        # make sure angular velocity in x-direction is zero
        np.testing.assert_array_almost_equal(av[:, 0], np.zeros(av[:, 0].shape))
        # make sure angular velocity in z-direction is zero
        np.testing.assert_array_almost_equal(av[:, 2], np.zeros(av[:, 2].shape))
        # make sure angular velocity in y-direction is higher in middle
        self.assertGreater(np.abs(av[4, 1]), av[0, 1])
        self.assertGreater(np.abs(av[4, 1]), av[8, 1])
        # make sure angular velocity in middle is greater than zero (right-hand rule)
        self.assertGreater(av[4, 1], 0)

        # turn around in xz plane the other way
        v = np.array([[-1., 0, 0], [-1, 0, 0], [-1, 0, 0],
                      [-1, 0, -1], [0, 0, -1], [1, 0, -1],
                      [1, 0, 0], [1, 0, 0], [1, 0, 0]])
        av = kinematics.angular_velocity(v, dt)

        self.assertFalse(np.any(np.isnan(av)))
        # make sure angular velocity in x-direction is zero
        np.testing.assert_array_almost_equal(av[:, 0], np.zeros(av[:, 0].shape))
        # make sure angular velocity in z-direction is zero
        np.testing.assert_array_almost_equal(av[:, 2], np.zeros(av[:, 2].shape))
        # make sure angular velocity in y-direction is higher in middle
        self.assertGreater(np.abs(av[4, 1]), av[0, 1])
        self.assertGreater(np.abs(av[4, 1]), av[8, 1])
        # make sure angular velocity in middle is less than zero (right-hand rule)
        self.assertLess(av[4, 1], 0)


class DistanceToWallTestCase(unittest.TestCase):

    def setUp(self):
        print("In test '{}'...".format(self._testMethodName))

    def test_example_distances_calculated_correctly(self):
        wall_bounds = ((-2, 4), (-1, 3), (-3, 4))

        positions = np.array([[-1., 1, 1],
                              [-2., 1, 1],
                              [-2., 0.5, 1],
                              [-3., 0.25, 1],
                              [-3, -0.5, 1]])

        d_correct = np.array([1, 0, 0, -1, -1])
        d = kinematics.distance_from_wall(positions, wall_bounds)
        np.testing.assert_array_almost_equal(d, d_correct)

        positions = np.array([[0., 0, 0],
                              [1, 0, 5],
                              [1, 0, 0],
                              [0, -2, 0],
                              [0, -2, -3]])

        d_correct = np.array([1., -1, 1, -1, -1])
        d = kinematics.distance_from_wall(positions, wall_bounds)
        np.testing.assert_array_almost_equal(d, d_correct)


class NormTestCase(unittest.TestCase):

    def setUp(self):
        print("In test '{}'...".format(self._testMethodName))

    def test_example_norm_calculated_and_ints_handled_correctly(self):
        v = np.array([[1, 1, 1],
                      [2, 3, 4],
                      [5, 1, -1]])

        n_correct = np.array([np.sqrt(3), 5.3851648071345037, 5.196152422706632])
        np.testing.assert_array_almost_equal(n_correct, kinematics.norm(v))

    def test_tiling_works_correctly(self):
        v = np.random.normal(0, 1, (100, 5))

        self.assertEqual(v.shape, kinematics.norm(v, '2darray').shape)


if __name__ == '__main__':
    unittest.main()