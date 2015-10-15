"""
Test to make sure calculate odor stats correctly add things to database.
"""
from __future__ import print_function, division
import numpy as np
import unittest

from db_api import models
from scripts.calculate_odor_stats import main, session


class MainTestCase(unittest.TestCase):

    def test_that_odor_stat_was_correctly_calculated(self):

        for expt in session.query(models.Experiment):
            integrated_odors = []
            for traj in session.query(models.Trajectory).filter_by(experiment=expt, clean=True):
                self.assertTrue(isinstance(traj.odor_stats.integrated_odor, float))
                integrated_odors.append(traj.odor_stats.integrated_odor)
            integrated_odors = np.array(integrated_odors)

            self.assertGreater(np.sum(integrated_odors > 0), 0)


if __name__ == '__main__':
    main()
    unittest.main()