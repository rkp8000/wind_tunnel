"""
Tests to make sure trajectories were cleaned correctly.
"""
from __future__ import print_function, division
import unittest
from db_api import models
from scripts.clean_trajectories import main, session, clean_traj

N_EXAMPLE_TRAJS_TO_CLEAN = 5


class CleanedTrajecsInDatabaseTestCase(unittest.TestCase):

    def test_some_but_not_all_raw_trajectories_are_clean(self):

        all_trajs = session.query(models.Trajectory).all()
        raw_not_clean = session.query(models.Trajectory).filter_by(raw=True, clean=False).all()
        clean_not_raw = session.query(models.Trajectory).filter_by(raw=False, clean=True).all()
        raw_and_clean = session.query(models.Trajectory).filter_by(raw=True, clean=True).all()
        not_raw_not_clean = session.query(models.Trajectory).filter_by(raw=False, clean=False).all()

        for traj_set in (all_trajs, raw_not_clean, clean_not_raw, raw_and_clean):
            self.assertGreater(len(traj_set), 0)
        self.assertEqual(len(not_raw_not_clean), 0)

        self.assertEqual(len(all_trajs), len(raw_not_clean) + \
                         len(clean_not_raw) + \
                         len(raw_and_clean) + \
                         len(not_raw_not_clean))

    def test_example_trajectories_were_cleaned_correctly(self):

        trajs_raw_not_clean = session.query(models.Trajectory).filter_by(raw=True, clean=False).all()

        for traj in trajs_raw_not_clean[:N_EXAMPLE_TRAJS_TO_CLEAN]:

            # get cleaning parameters
            insect = traj.experiment.insect
            cleaning_params_list = session.query(models.TrajectoryCleaningParameter.param,
                                             models.TrajectoryCleaningParameter.value).\
                                             filter_by(insect=insect).all()
            cleaning_params = dict(cleaning_params_list)

            # clean traj
            clean_portions = clean_traj(traj, cleaning_params)

            # check to make sure the correct number of clean trajectories were generated with the right ids
            # and that they have the right timepoint indicators
            for ctr, clean_portion in enumerate(clean_portions):

                clean_id = traj.id + '_c{}'.format(ctr)
                traj_cleaned = session.query(models.Trajectory).get(clean_id)

                self.assertFalse(traj_cleaned is None)
                self.assertEqual(traj_cleaned.start_timepoint_id, clean_portion[0])
                self.assertEqual(traj_cleaned.end_timepoint_id, clean_portion[1])

            # make sure that there is no additional trajectory that was created
            self.assertTrue(session.query(models.Trajectory). \
                            get(traj.id + '_c{}'.format(len(clean_portions))) is None)


if __name__ == '__main__':
    print("Running 'main()'...")
    main()
    print("Running tests...")
    unittest.main()