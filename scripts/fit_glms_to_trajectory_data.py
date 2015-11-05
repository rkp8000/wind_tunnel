"""
Script for fitting a set of models to a set of trajectories.
"""
from __future__ import division, print_function
import numpy as np

import fitting
import insect_glm_fit_helper as igfh

from db_api import models
from db_api.connect import session, commit

DATA_DIR_ENV_VAR = 'WIND_TUNNEL_FIGURE_DATA_DIRECTORY'

SAVE_DIR = 'glm_fit'

FIT_NAME = 'short_long_timescale_bases_fit_0'

EXPERIMENT_IDS = [
    'fruitfly_0.3mps_checkerboard_floor',
    'fruitfly_0.4mps_checkerboard_floor',
    'fruitfly_0.6mps_checkerboard_floor',
    'mosquito_0.4mps_checkerboard_floor',
]

ODOR_STATES = [
    'on',
    'none',
    'afterodor',
]

N_TRIALS = 100
N_TRAIN = 400
N_TEST = 100

INTEGRATED_ODOR_THRESHOLD = 10

LINK = 'identity'
FAMILY = 'Gaussian'

PREDICTED = 'heading_xyz'

INPUT_SETS = [
    (),
    ('position_x', ),
    ('position_x', 'odor'),
    ('position_x', 'odor'),
    ('position_x', 'odor')
]

OUTPUTS = [PREDICTED] * len(INPUT_SETS)

DELAY = 1

INPUT_TAUS = [  # in units of timesteps (dt = 0.01s)
    (),
    (None, ),
    (None, None),
    (None, [2, 5, 7, 10, 25]),
    (None, [2, 5, 7, 10, 25, 150, 250]),
]

OUTPUT_TAUS = [
    [2, 5, 7, 10, 25],
    [2, 5, 7, 10, 25],
    [2, 5, 7, 10, 25],
    [2, 5, 7, 10, 25],
    [2, 5, 7, 10, 25],
]

START_TIMEPOINT = 20
DOMAIN_FACTOR = 4.61


def main(n_trials, n_train_max, n_test_max, root_dir_env_var):

    # make basis functions
    basis_ins, basis_outs, max_filter_length = igfh.make_exponential_basis_functions(
        INPUT_TAUS, OUTPUT_TAUS, DOMAIN_FACTOR
    )

    for expt_id in EXPERIMENT_IDS:
        for odor_state in ODOR_STATES:

            trajs = igfh.get_trajs_with_integrated_odor_above_threshold(
                expt_id, odor_state, INTEGRATED_ODOR_THRESHOLD
            )

            train_test_ratio = (n_train_max / (n_train_max + n_test_max))
            test_train_ratio = (n_test_max / (n_train_max + n_test_max))
            n_train = min(n_train_max, np.floor(len(trajs) * train_test_ratio))
            n_test = min(n_test_max, np.floor(len(trajs) * test_train_ratio))

            trajs_trains = []
            trajs_tests = []
            glmss = []
            residualss = []

            for trial_ctr in range(n_trials):
                print('{}: odor {} (trial number: {})'.format(expt_id, odor_state, trial_ctr))

                # get random set of training and test trajectories
                perm = np.random.permutation(len(trajs))
                train_idxs = perm[:n_train]
                test_idxs = perm[-n_test:]

                trajs_train = list(np.array(trajs)[train_idxs])
                trajs_test = list(np.array(trajs)[test_idxs])

                # do some more stuff
                glms = []
                residuals = []
                for input_set, output, basis_in, basis_out in zip(INPUT_SETS, OUTPUTS, basis_ins, basis_outs):

                    # get relevant time-series data from each trajectory set
                    data_train = igfh.time_series_from_trajs(
                        trajs_train,
                        inputs=input_set,
                        output=output
                    )
                    data_test = igfh.time_series_from_trajs(
                        trajs_test,
                        inputs=input_set,
                        output=output
                    )

                    glm = fitting.GLMFitter(link=LINK, family=FAMILY)
                    glm.set_params(DELAY, basis_in=basis_in, basis_out=False)

                    glm.input_set = input_set
                    glm.output = output

                    # fit to training data
                    glm.fit(data=data_train, start=START_TIMEPOINT)

                    # predict test data
                    prediction = glm.predict(data=data_test, start=START_TIMEPOINT)
                    _, ground_truth = glm.make_feature_matrix_and_response_vector(data_test, START_TIMEPOINT)

                    # calculate residual
                    residual = np.sqrt(((prediction - ground_truth)**2).mean())

                    # clear out feature matrix and response from glm for efficient storage
                    glm.feature_matrix = None
                    glm.response_vector = None
                    glm.results.remove_data()
                    # store things
                    glms.append(glm)
                    residuals.append(residual)

                trajs_train_ids = [traj.id for traj in trajs_train]
                trajs_test_ids = [traj.id for traj in trajs_test]
                trajs_trains.append(trajs_train_ids)
                trajs_tests.append(trajs_test_ids)
                glmss.append(glms)
                residualss.append(residuals)

            # save a glm fit set
            glm_fit_set = models.GlmFitSet()

            # add data to it
            glm_fit_set.root_dir_env_var = root_dir_env_var
            glm_fit_set.path_relative = 'glm_fit'
            glm_fit_set.file_name = '{}_{}_odor_{}.pickle'.format(FIT_NAME, expt_id, odor_state)
            glm_fit_set.experiment = session.query(models.Experiment).get(expt_id)
            glm_fit_set.odor_state = odor_state
            glm_fit_set.name = FIT_NAME
            glm_fit_set.link = LINK
            glm_fit_set.family = FAMILY
            glm_fit_set.integrated_odor_threshold = INTEGRATED_ODOR_THRESHOLD
            glm_fit_set.predicted = PREDICTED
            glm_fit_set.delay = DELAY
            glm_fit_set.start_time_point = START_TIMEPOINT
            glm_fit_set.n_glms = len(glms)
            glm_fit_set.n_train = n_train
            glm_fit_set.n_test = n_test
            glm_fit_set.n_trials = n_trials

            # save data file
            glm_fit_set.save_to_file(
                input_sets=INPUT_SETS,
                outputs=OUTPUTS,
                basis_in=basis_ins,
                basis_out=basis_outs,
                trajs_train=trajs_trains,
                trajs_test=trajs_tests,
                glms=glmss,
                residuals=residualss
            )

            # save everything else (+ link to data file) in database
            session.add(glm_fit_set)

            commit(session)


if __name__ == '__main__':
    main(N_TRIALS, N_TRAIN, N_TEST, DATA_DIR_ENV_VAR)