"""
Unit tests for calculating basic crossing features.
These include position at entry, peak, and exit of crossing and heading at entry, peak, and exit.
"""
from __future__ import print_function, division
import numpy as np
import unittest
from db_api import models
from scripts.calculate_crossing_features_basic import main, session, DETERMINATION

N_EXAMPLE_CROSSINGS = 5


class MainTestCase(unittest.TestCase):

    def setUp(self):

        self.expts = session.query(models.Experiment)

    def test_one_basic_feature_set_per_crossing_calculated(self):

        for expt in self.expts:

            # loop through crossing groups with this determined threshold
            threshold = session.query(models.Threshold).filter_by(determination=DETERMINATION,
                                                                  experiment=expt).first()
            for cg in threshold.crossing_groups:
                for crossing in cg.crossings:
                    self.assertTrue(crossing.feature_set_basic is not None)

    def test_example_crossings_basic_feature_set_calculated_correctly(self):

        for expt in self.expts:

            # loop through crossing groups with this determined threshold
            threshold = session.query(models.Threshold).filter_by(determination=DETERMINATION,
                                                                  experiment=expt).first()
            for cg in threshold.crossing_groups:
                for crossing in cg.crossings[:N_EXAMPLE_CROSSINGS]:
                    # check that positions are correct
                    position_x_entry = crossing.timepoint_field(session, 'position_x', 0, 0, 'entry', 'entry')[0]
                    position_y_entry = crossing.timepoint_field(session, 'position_y', 0, 0, 'entry', 'entry')[0]
                    position_z_entry = crossing.timepoint_field(session, 'position_z', 0, 0, 'entry', 'entry')[0]

                    position_x_peak = crossing.timepoint_field(session, 'position_x', 0, 0, 'peak', 'peak')[0]
                    position_y_peak = crossing.timepoint_field(session, 'position_y', 0, 0, 'peak', 'peak')[0]
                    position_z_peak = crossing.timepoint_field(session, 'position_z', 0, 0, 'peak', 'peak')[0]

                    position_x_exit = crossing.timepoint_field(session, 'position_x', 0, 0, 'exit', 'exit')[0]
                    position_y_exit = crossing.timepoint_field(session, 'position_y', 0, 0, 'exit', 'exit')[0]
                    position_z_exit = crossing.timepoint_field(session, 'position_z', 0, 0, 'exit', 'exit')[0]

                    heading_xy_entry = crossing.timepoint_field(session, 'heading_xy', 0, 0, 'entry', 'entry')[0]
                    heading_xz_entry = crossing.timepoint_field(session, 'heading_xz', 0, 0, 'entry', 'entry')[0]
                    heading_xyz_entry = crossing.timepoint_field(session, 'heading_xyz', 0, 0, 'entry', 'entry')[0]

                    heading_xy_peak = crossing.timepoint_field(session, 'heading_xy', 0, 0, 'peak', 'peak')[0]
                    heading_xz_peak = crossing.timepoint_field(session, 'heading_xz', 0, 0, 'peak', 'peak')[0]
                    heading_xyz_peak = crossing.timepoint_field(session, 'heading_xyz', 0, 0, 'peak', 'peak')[0]

                    heading_xy_exit = crossing.timepoint_field(session, 'heading_xy', 0, 0, 'exit', 'exit')[0]
                    heading_xz_exit = crossing.timepoint_field(session, 'heading_xz', 0, 0, 'exit', 'exit')[0]
                    heading_xyz_exit = crossing.timepoint_field(session, 'heading_xyz', 0, 0, 'exit', 'exit')[0]

                    self.assertAlmostEqual(position_x_entry, crossing.feature_set_basic.position_x_entry)
                    self.assertAlmostEqual(position_y_entry, crossing.feature_set_basic.position_y_entry)
                    self.assertAlmostEqual(position_z_entry, crossing.feature_set_basic.position_z_entry)

                    self.assertAlmostEqual(position_x_peak, crossing.feature_set_basic.position_x_peak)
                    self.assertAlmostEqual(position_y_peak, crossing.feature_set_basic.position_y_peak)
                    self.assertAlmostEqual(position_z_peak, crossing.feature_set_basic.position_z_peak)

                    self.assertAlmostEqual(position_x_exit, crossing.feature_set_basic.position_x_exit)
                    self.assertAlmostEqual(position_y_exit, crossing.feature_set_basic.position_y_exit)
                    self.assertAlmostEqual(position_z_exit, crossing.feature_set_basic.position_z_exit)

                    self.assertAlmostEqual(heading_xy_entry, crossing.feature_set_basic.heading_xy_entry)
                    self.assertAlmostEqual(heading_xz_entry, crossing.feature_set_basic.heading_xz_entry)
                    self.assertAlmostEqual(heading_xyz_entry, crossing.feature_set_basic.heading_xyz_entry)

                    self.assertAlmostEqual(heading_xy_peak, crossing.feature_set_basic.heading_xy_peak)
                    self.assertAlmostEqual(heading_xz_peak, crossing.feature_set_basic.heading_xz_peak)
                    self.assertAlmostEqual(heading_xyz_peak, crossing.feature_set_basic.heading_xyz_peak)

                    self.assertAlmostEqual(heading_xy_exit, crossing.feature_set_basic.heading_xy_exit)
                    self.assertAlmostEqual(heading_xz_exit, crossing.feature_set_basic.heading_xz_exit)
                    self.assertAlmostEqual(heading_xyz_exit, crossing.feature_set_basic.heading_xyz_exit)

                    # make sure these are the same as would be calculated by directly looking up the
                    # timepoint value given the timepoint id

                    entry_tp_id = crossing.entry_timepoint_id
                    peak_tp_id = crossing.peak_timepoint_id
                    exit_tp_id = crossing.exit_timepoint_id

                    # just look up a few values
                    position_x_entry = session.query(models.Timepoint.position_x).filter_by(id=entry_tp_id).first()[0]
                    heading_xy_peak = session.query(models.Timepoint.heading_xy).filter_by(id=peak_tp_id).first()[0]
                    heading_xyz_exit = session.query(models.Timepoint.heading_xyz).filter_by(id=exit_tp_id).first()[0]

                    self.assertAlmostEqual(position_x_entry, crossing.feature_set_basic.position_x_entry)
                    self.assertAlmostEqual(heading_xy_peak, crossing.feature_set_basic.heading_xy_peak)
                    self.assertAlmostEqual(heading_xyz_exit, crossing.feature_set_basic.heading_xyz_exit)


if __name__ == '__main__':
    main()
    unittest.main()