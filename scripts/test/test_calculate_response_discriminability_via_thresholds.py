"""
Tests for calculate_response_discriminability_via_thresholds.py.
"""
from __future__ import print_function, division
import numpy as np
import unittest
from db_api import models
from scripts.calculate_response_discriminability_via_thresholds \
    import main, session, DISCRIMINATION_THRESHOLD_VALUES, TIME_AVG_START, TIME_AVG_END, \
    TIME_AVG_REL_TO, RESPONSE_VAR


class MainTestCase(unittest.TestCase):

    def setUp(self):

        self.cgs = session.query(models.CrossingGroup).\
            filter(models.CrossingGroup.odor_state == 'on').\
            filter(models.Threshold.determination == 'arbitrary')

    def test_each_crossing_group_has_correct_number_of_discrimination_thresholds(self):

        for cg in self.cgs:

            correct_number = len(DISCRIMINATION_THRESHOLD_VALUES[cg.experiment.insect])
            self.assertEqual(len(cg.discrimination_thresholds), correct_number)

    def test_discrimination_thresholds_have_reasonable_numbers(self):

        for cg in self.cgs:
            for disc_th in cg.discrimination_thresholds:

                self.assertEqual(disc_th.time_avg_start, TIME_AVG_START)
                self.assertEqual(disc_th.time_avg_end, TIME_AVG_END)
                self.assertEqual(disc_th.variable, RESPONSE_VAR)
                self.assertEqual(disc_th.time_avg_rel_to, TIME_AVG_REL_TO)

                if disc_th.n_crossings_below == 0 or disc_th.n_crossings_above == 0:
                    self.assertEqual(disc_th.time_avg_difference, None)
                    self.assertEqual(disc_th.lower_bound, None)
                    self.assertEqual(disc_th.upper_bound, None)
                elif disc_th.n_crossings_below == 1 or disc_th.n_crossings_above == 1:
                    self.assertNotEqual(disc_th.time_avg_difference, None)
                    self.assertEqual(disc_th.lower_bound, None)
                    self.assertEqual(disc_th.upper_bound, None)
                else:
                    self.assertGreater(disc_th.time_avg_difference, disc_th.lower_bound)
                    self.assertLess(disc_th.time_avg_difference, disc_th.upper_bound)


if __name__ == '__main__':
    main()
    unittest.main()