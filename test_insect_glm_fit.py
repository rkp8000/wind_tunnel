"""
Test/demo for fitting actual insect data with glms to determine appropriate length scale of filters.
"""
from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import os
import statsmodels.api as sm
import unittest

import fitting
import insect_glm_fit_helper as igfh

N_TRAIN = 100
N_TEST = 50

EXPT_ID = 'fruitfly_0.4mps_checkerboard_floor'
ODOR_STATE = 'on'

LINK_NAME = 'identity'
FAMILY_NAME = 'Gaussian'
PREDICTED_VARIABLE = 'heading'

INTEGRATED_ODOR_THRESHOLD = 10

DELAY = 5  # in timesteps (dt = 0.01s)

SAVE_DIR = '/Users/rkp/Desktop'

INPUT_SETS = [
    (),
    ('position_x', ),
    ('position_x', 'odor'),
    ('position_x', 'odor'),
    ('position_x', 'odor')
]

OUTPUT = 'heading_xyz'
OUTPUTS = [OUTPUT] * len(INPUT_SETS)

INPUT_TAUS = [  # in units of time points (not seconds)
    (),
    (None, ),
    (None, None),
    (None, [2, 5, 7, 10, 20]),
    (None, [2, 5, 7, 10, 20, 60, 80]),
]

OUTPUT_TAUS = [
    [2, 5, 7, 10, 20],
    [2, 5, 7, 10, 20],
    [2, 5, 7, 10, 20],
    [2, 5, 7, 10, 20],
    [2, 5, 7, 10, 20],
]

START_TIMEPOINT = 20

DOMAIN_FACTOR = 4.61  # x s.t., e^-x = 0.01


class VariousModelsDemo(unittest.TestCase):

    def test_fitting_of_multiple_models_to_single_training_set_and_seeing_how_well_they_predict_test_set(self):

        # make basis sets for each model
        print('Making filter basis functions...')
        basis_ins = []  # i.e., one for each model
        basis_outs = []

        for input_tau, output_tau in zip(INPUT_TAUS, OUTPUT_TAUS):
            basis_in = []
            for tau_set in input_tau:
                if tau_set is None:
                    basis = None
                else:
                    t = np.arange(np.round(DOMAIN_FACTOR * np.max(tau_set)), dtype=float)
                    basis = np.zeros((len(t), len(tau_set)), dtype=float)
                    for ctr, tau in enumerate(tau_set):
                        basis[:, ctr] = np.exp(-t / tau)
                basis_in.append(basis)

            if output_tau is None:
                basis_out = None
            else:
                t = np.arange(np.round(DOMAIN_FACTOR * np.max(output_tau)), dtype=float)
                basis_out = np.zeros((len(t), len(output_tau)), dtype=float)
                for ctr, tau in enumerate(output_tau):
                    basis_out[:, ctr] = np.exp(-t / tau)

            basis_ins.append(basis_in)
            basis_outs.append(basis_out)

        n_models = len(basis_outs)

        # figure out max filter length
        max_filter_length = 0
        for basis_in, basis_out in zip(basis_ins, basis_outs):
            in_filter_max = np.max([0] + [len(basis) for basis in basis_in if basis is not None])
            if basis_out is not None:
                out_filter_max = len(basis_out)
            else:
                out_filter_max = 0
            filter_max = max(in_filter_max, out_filter_max)
            if filter_max > max_filter_length:
                max_filter_length = filter_max
        print('Max filter length is {}'.format(max_filter_length))

        print('Getting trajectories...')
        trajs = igfh.get_trajs_with_integrated_odor_above_threshold(
            experiment_id=EXPT_ID,
            odor_state=ODOR_STATE,
            integrated_odor_threshold=INTEGRATED_ODOR_THRESHOLD
        )
        # split these into training and test trajectories
        trajs_train = trajs[:N_TRAIN]
        trajs_test = trajs[N_TRAIN:N_TRAIN + N_TEST]

        if LINK_NAME == 'identity':
            link_function = sm.genmod.families.links.identity
        elif LINK_NAME == 'log':
            link_function = sm.genmod.families.links.log
        if FAMILY_NAME == 'Gaussian':
            family = sm.families.Gaussian(link=link_function)

        # fit each of N models to training data and predict test data
        print('Fitting models...')
        models = []
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

            model = fitting.GLMFitter(family)
            model.set_params(DELAY, basis_in=basis_in, basis_out=basis_out)

            # fit to training data
            model.fit(data=data_train, start=START_TIMEPOINT)

            # predict test data
            prediction = model.predict(data=data_test, start=START_TIMEPOINT)
            _, ground_truth = model.make_feature_matrix_and_response_vector(data_test, START_TIMEPOINT)

            # calculate residual
            residual = np.sqrt(((prediction - ground_truth)**2).mean())

            # store things
            models.append(model)
            residuals.append(residual)

        print('Generating plots...')
        # plot basis and filters for each model, as well as example time-series with prediction
        fig_filt, axs_filt = plt.subplots(
            n_models, 2, facecolor='white', figsize=(10, 10), tight_layout=True
        )
        fig_ts, axs_ts = plt.subplots(
            n_models, 1, facecolor='white', figsize=(10, 10), tight_layout=True
        )
        for model, res, ax_filt_row, ax_ts in zip(models, residuals, axs_filt, axs_ts):

            model.plot_filters(ax_filt_row[0], x_lim=(0, 100))
            model.plot_basis(ax_filt_row[1], x_lim=(0, 100))
            prediction_0 = model.predict(data=data_test[0:1], start=START_TIMEPOINT)
            _, ground_truth_0 = model.make_feature_matrix_and_response_vector(data_test[0:1], START_TIMEPOINT)

            t = np.arange(len(data_test[0][1]))[-len(prediction_0):]
            print(t[0])
            ax_ts.plot(t, ground_truth_0, color='k', ls='-')
            ax_ts.plot(t, prediction_0, color='r', ls='--')

            ax_filt_row[0].set_ylabel('filter\nstrength')
            ax_ts.set_title('Residual = {}'.format(res))

        axs_filt[-1][0].set_xlabel('timestep')
        axs_filt[-1][1].set_xlabel('timestep')
        axs_ts[-1].set_xlabel('timestep')

        fig_filt.savefig(os.path.join(SAVE_DIR, 'filters.png'))
        fig_ts.savefig(os.path.join(SAVE_DIR, 'example_predictions.png'))

        plt.show()


if __name__ == '__main__':
    unittest.main()