"""
Tests to make sure low threshold crossings were stored correctly in database.
"""
from __future__ import print_function, division
import numpy as np
import unittest
from db_api import models
from scripts.identify_low_threshold_crossings import main, time_series, session

ODOR_STATES = ('on', 'none', 'afterodor')
LOW_THRESHOLD_VALUES = {'fruit_fly': (0.01, 0.1), 'mosquito': (401, 410)}
DTH = .0001
N_CROSSINGS_TO_CHECK_PER_CROSSING_GROUP = 3


class MainTestCase(unittest.TestCase):

    def setUp(self):
        print("In test '{}'...".format(self._testMethodName))
        self.expts = session.query(models.Experiment)

    def test_crossing_groups_were_stored_correctly(self):

        for expt in self.expts:
            # make sure each experiment has one of each of the two thresholds
            # and that each threshold has three crossing groups, one for each odor state
            low_threshold_values = LOW_THRESHOLD_VALUES[expt.insect]

            for low_threshold_value in low_threshold_values:
                thresholds = session.query(models.Threshold).\
                    filter(models.Threshold.experiment == expt).\
                    filter(models.Threshold.determination == 'arbitrary').\
                    filter(models.Threshold.value.between(low_threshold_value - DTH,
                                                          low_threshold_value + DTH)).all()
                self.assertEqual(len(thresholds), 1)

                self.assertEqual(len(thresholds[0].crossing_groups), 3)
                odor_states = [cg.odor_state for cg in thresholds[0].crossing_groups]
                self.assertEqual(set(odor_states), set(ODOR_STATES))

                for cg in thresholds[0].crossing_groups:
                    self.assertGreater(len(cg.crossings), 0)

    def test_example_crossings_were_computed_correctly(self):

        thresholds = session.query(models.Threshold).filter_by(determination='arbitrary').all()

        for threshold in thresholds:
            for cg in threshold.crossing_groups:
                crossings = list(cg.crossings)
                for _ in range(N_CROSSINGS_TO_CHECK_PER_CROSSING_GROUP):
                    traj = np.random.choice(crossings).trajectory

                    # recompute this trajectory's crossings
                    segments, peaks = time_series.segment_by_threshold(traj.odors(session),
                                                                       threshold.value,
                                                                       traj.timepoint_ids_extended)

                    traj_crossings = session.query(models.Crossing).\
                        filter_by(trajectory=traj, crossing_group=cg).\
                        order_by(models.Crossing.crossing_number)

                    for ctr, (segment, peak) in enumerate(zip(segments, peaks)):
                        self.assertEqual(segment[0], traj_crossings[ctr].start_timepoint_id)
                        self.assertEqual(segment[1], traj_crossings[ctr].entry_timepoint_id)
                        self.assertEqual(segment[2], traj_crossings[ctr].peak_timepoint_id)
                        self.assertEqual(segment[3], traj_crossings[ctr].exit_timepoint_id + 1)
                        self.assertEqual(segment[4], traj_crossings[ctr].end_timepoint_id + 1)
                        self.assertAlmostEqual(peak, traj_crossings[ctr].max_odor)


if __name__ == '__main__':
    # run main script
    main()
    # run tests
    unittest.main()