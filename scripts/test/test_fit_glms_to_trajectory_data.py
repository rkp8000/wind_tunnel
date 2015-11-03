"""
Run basic test of fit_glms_to_trajectory_data.py.
"""
from __future__ import print_function, division
import numpy as np
import unittest

import insect_glm_fit_helper as igfh
from db_api import models
from scripts import fit_glms_to_trajectory_data as script

N_TRAIN = 2
N_TEST = 2
N_TRIALS = 2

DATA_DIR_ENV_VAR_TEST = 'WIND_TUNNEL_TEST_FIGURE_DATA_DIRECTORY'


class MainTestCase(unittest.TestCase):

    def test_data_was_stored_correctly(self):

        # loop through all experiments and odor states and make sure a GlmFitSet was stored with correct info
        for expt_id in script.EXPERIMENT_IDS:
            for odor_state in script.ODOR_STATES:

                glm_fit_set = script.session.query(models.GlmFitSet).filter_by(
                    experiment_id=expt_id,
                    odor_state=odor_state,
                ).first()

                # make sure stuff was saved to database
                self.assertEqual(glm_fit_set.name, script.FIT_NAME)
                self.assertEqual(glm_fit_set.link, script.LINK)
                self.assertEqual(glm_fit_set.family, script.FAMILY)
                self.assertAlmostEqual(glm_fit_set.integrated_odor_threshold, script.INTEGRATED_ODOR_THRESHOLD)
                self.assertEqual(glm_fit_set.predicted, script.PREDICTED)
                self.assertEqual(glm_fit_set.delay, script.DELAY)
                self.assertEqual(glm_fit_set.start_time_point, script.START_TIMEPOINT)
                self.assertEqual(glm_fit_set.n_glms, len(script.INPUT_SETS))
                self.assertEqual(glm_fit_set.n_trials, N_TRIALS)

                self.assertEqual(glm_fit_set.input_sets, script.INPUT_SETS)
                self.assertEqual(glm_fit_set.outputs, script.OUTPUTS)
                self.assertEqual(len(glm_fit_set.input_sets), len(glm_fit_set.outputs))
                self.assertEqual(len(glm_fit_set.basis_in), len(script.INPUT_SETS))
                self.assertEqual(len(glm_fit_set.basis_out), len(glm_fit_set.outputs))

                self.assertEqual(len(glm_fit_set.trajs_train), N_TRIALS)
                self.assertEqual(len(glm_fit_set.trajs_test), N_TRIALS)

                self.assertEqual(len(glm_fit_set.glms), N_TRIALS)
                self.assertEqual(len(glm_fit_set.residuals), N_TRIALS)

                self.assertGreater(glm_fit_set.residuals[-1], 0)

                self.assertTrue(isinstance(glm_fit_set.glms[0][0], script.fitting.GLMFitter))

    def test_that_residual_calculations_were_done_correctly(self):

        for expt_id in script.EXPERIMENT_IDS:
            for odor_state in script.ODOR_STATES:

                glm_fit_set = script.session.query(models.GlmFitSet).filter_by(
                    experiment_id=expt_id,
                    odor_state=odor_state,
                ).first()

                # make sure predictions residuals add up to true residuals
                glm = glm_fit_set.glms[-1][-1]
                start = glm_fit_set.start_time_point
                residual_stored = glm_fit_set.residuals[-1][-1]
                data_test = igfh.time_series_from_trajs(
                    glm_fit_set.trajs_test[-1],
                    inputs=glm.input_set,
                    output=glm.output
                )

                prediction = glm.predict(data=data_test, start=start)
                _, ground_truth = glm.make_feature_matrix_and_response_vector(data_test, start)

                residual_recalculated = np.sqrt(((prediction - ground_truth)**2).mean())

                self.assertAlmostEqual(residual_stored, residual_recalculated)


if __name__ == '__main__':
    #script.main(N_TRIALS, N_TRAIN, N_TEST, DATA_DIR_ENV_VAR_TEST)
    unittest.main()