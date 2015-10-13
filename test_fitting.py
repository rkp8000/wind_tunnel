"""
Tests to make sure we can correctly fit time-series whose construction details we have ground truth for.
"""
from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import ndimage
import statsmodels.api as sm
import unittest

import time_series

FIGURE_SAVE_DIR = '/Users/rkp/Desktop'


class ModelComparisonTestCase(unittest.TestCase):

    def test_recovery_of_filters_from_basic_ARMA_fit(self):
        """
        Here we create an output that is a filtered version of an input that uses filters at two time-scales.
        We show that when filters at only one time-scale are allowed to be fit, the prediction error is worse
        than when filters are allowed to be of the longer time-scale.
        """
        # make our signals
        T = 500
        NOISE = 0.05
        SCALING = 0.2
        DELAY = 5
        cc = np.concatenate

        in_1 = ndimage.gaussian_filter1d(np.random.normal(0, 1, (T,)), 1)
        in_2 = ndimage.gaussian_filter1d(np.random.normal(0, 2, (T,)), 3)

        c = -0.4
        t = np.linspace(0, 100)
        b_1 = np.exp(-t/5)/5
        b_2 = np.exp(-t/10)/10
        b_3 = np.exp(-t/15)/15
        b_4 = np.exp(-t/50)/50
        b_5 = np.exp(-t/60)/60
        b_6 = np.exp(-t/70)/70

        b = np.array([b_1, b_2, b_3, b_4, b_5, b_6]).T

        f_in_1 = b.dot(SCALING*np.array([1, 1, 3, 0, 20, 0])[:, None])
        f_in_2 = b.dot(SCALING*np.array([-1, 3, -1, 0, 20, 0])[:, None])
        f_out = b.dot(SCALING*np.array([.5, 4, -.1, 0, 0, 0])[:, None])

        # create filtered output stimulus
        out = np.zeros((T,), dtype=float)
        f_len = len(f_out)  # filter length

        # go through each timestep from delay to end (zero padding early ones) and compute output
        # based on filtered input and filtered history
        for ts in range(DELAY, T):
            if ts < DELAY + f_len:
                in_1_subset = cc([np.zeros((f_len - ts + DELAY - 1,), dtype=float), in_1[:ts - DELAY + 1]])
                in_2_subset = cc([np.zeros((f_len - ts + DELAY - 1,), dtype=float), in_2[:ts - DELAY + 1]])
                out_subset = cc([np.zeros((f_len - ts + DELAY - 1,), dtype=float), out[:ts - DELAY + 1]])
            else:
                in_1_subset = in_1[ts - DELAY + 1 - f_len:ts - DELAY + 1]
                in_2_subset = in_2[ts - DELAY + 1 - f_len:ts - DELAY + 1]
                out_subset = out[ts - DELAY + 1 - f_len:ts - DELAY + 1]

            out[ts] = in_1_subset.dot(f_in_1[::-1]) + in_2_subset.dot(f_in_2[::-1]) + out_subset.dot(f_out[::-1]) + c
            out[ts] += np.random.normal(0, NOISE)

        # munge time-series data so we can pass it to statsmodels glm fitter
        tsm = time_series.Munger()
        tsm.delay = DELAY
        tsm.basis_in = [b, b]
        tsm.basis_out = b
        tsm.orthogonalize_basis()
        feature_matrix, response_vector = tsm.munge([in_1, in_2], out, start=f_len)

        # make sure the shapes are correct (for good luck!)
        self.assertEqual(feature_matrix.shape[0], T - f_len - DELAY)
        self.assertEqual(feature_matrix.shape[1], 19)
        self.assertEqual(len(response_vector), T - f_len - DELAY)

        # fit an ordinary least squares model with statsmodels
        link_function = sm.genmod.families.links.identity
        family = sm.families.Gaussian(link=link_function)
        model = sm.GLM(endog=response_vector, exog=feature_matrix, family=family)
        results = model.fit()
        prediction = model.predict(results.params, exog=feature_matrix)

        # reconstruct filters from coefficients
        constant, in_filters, out_filter = tsm.filters_from_coeffs(results.params)

        print('True constant: {}'.format(c))
        print('Recovered constant: {}'.format(constant))

        # make figure to output test results
        fig = plt.figure(figsize=(10, 5), tight_layout=True)
        ax_ts = fig.add_subplot(2, 1, 1)
        ax_filt_0 = fig.add_subplot(2, 2, 3)
        ax_filt_1 = fig.add_subplot(2, 2, 4, sharey=ax_filt_0)
        ax_filts = [ax_filt_0, ax_filt_1]

        ax_ts.plot(np.transpose([in_1, in_2, out]))
        ax_ts.plot(np.arange(T)[-len(prediction):], prediction, 'r--')
        ax_ts.set_xlabel('t')
        ax_ts.set_ylabel('input and output')

        ax_filts[0].plot(t, f_in_1, 'b')
        ax_filts[0].plot(t, f_in_2, 'g')
        ax_filts[0].plot(t, f_out, 'r')
        ax_filts[0].set_xlim(t[0], t[-1])
        ax_filts[0].set_xlabel('t')
        ax_filts[0].set_ylabel('filter strength')
        ax_filts[0].set_title('true filters')

        ax_filts[1].plot(t, in_filters[0], 'b')
        ax_filts[1].plot(t, in_filters[1], 'g')
        ax_filts[1].plot(t, out_filter, 'r')
        ax_filts[1].set_xlim(t[0], t[-1])
        ax_filts[1].set_xlabel('t')
        ax_filts[1].set_ylabel('filter strength')
        ax_filts[1].set_title('recovered filters')

        SAVE_PATH = os.path.join(
            FIGURE_SAVE_DIR,
            'test_recovery_of_filters_from_basic_ARMA_fit.png'
        )
        fig.savefig(SAVE_PATH)

        print('Figure saved at {}'.format(SAVE_PATH))

    def test_prediction_error_decreases_when_filters_are_allowed_to_take_on_proper_timescale(self):
        """
        Here we create an output that is a filtered version of an input that uses filters at two time-scales.
        We show that when filters at only one time-scale are allowed to be fit, the prediction error is worse
        than when filters are allowed to be of the longer time-scale.
        """
        # make our signals
        T = 500
        NOISE = 0.2
        SCALING = 0.2
        DELAY = 5
        cc = np.concatenate

        confound_1 = ndimage.gaussian_filter1d(np.random.normal(0, 1, (T,)), 7)
        in_11 = ndimage.gaussian_filter1d(np.random.normal(0, 1, (T,)), 1)
        in_12 = ndimage.gaussian_filter1d(np.random.normal(0, 2, (T,)), 3)

        confound_2 = ndimage.gaussian_filter1d(np.random.normal(0, 1, (T,)), 7)
        in_21 = ndimage.gaussian_filter1d(np.random.normal(0, 1, (T,)), 1)
        in_22 = ndimage.gaussian_filter1d(np.random.normal(0, 2, (T,)), 3)

        c = -0.4
        c_confound = 0.2
        t = np.linspace(0, 100)
        b_1 = np.exp(-t/5)/5
        b_2 = np.exp(-t/10)/10
        b_3 = np.exp(-t/15)/15
        b_4 = np.exp(-t/50)/50
        b_5 = np.exp(-t/60)/60
        b_6 = np.exp(-t/70)/70

        b = np.array([b_1, b_2, b_3, b_4, b_5, b_6]).T

        f_in_1 = b.dot(SCALING*np.array([1, 1, 3, 0, 20, 0])[:, None])
        f_in_2 = b.dot(SCALING*np.array([-1, 3, -1, 0, 20, 0])[:, None])
        f_out = b.dot(SCALING*np.array([.5, 4, -.1, 0, 0, 0])[:, None])

        # create filtered output stimulus
        out_1 = np.zeros((T,), dtype=float)
        out_2 = np.zeros((T,), dtype=float)
        f_len = len(f_out)  # filter length

        # go through each timestep from delay to end (zero padding early ones) and compute output
        # based on filtered input and filtered history

        # make first signal
        for ts in range(DELAY, T):
            if ts < DELAY + f_len:
                in_1_subset = cc([np.zeros((f_len - ts + DELAY - 1,), dtype=float), in_11[:ts - DELAY + 1]])
                in_2_subset = cc([np.zeros((f_len - ts + DELAY - 1,), dtype=float), in_12[:ts - DELAY + 1]])
                out_subset = cc([np.zeros((f_len - ts + DELAY - 1,), dtype=float), out_1[:ts - DELAY + 1]])
            else:
                in_1_subset = in_11[ts - DELAY + 1 - f_len:ts - DELAY + 1]
                in_2_subset = in_12[ts - DELAY + 1 - f_len:ts - DELAY + 1]
                out_subset = out_1[ts - DELAY + 1 - f_len:ts - DELAY + 1]

            out_1[ts] = in_1_subset.dot(f_in_1[::-1]) + in_2_subset.dot(f_in_2[::-1]) + \
                out_subset.dot(f_out[::-1]) + c + c_confound * confound_1[ts - DELAY] + np.random.normal(0, NOISE)

        for ts in range(DELAY, T):
            if ts < DELAY + f_len:
                in_1_subset = cc([np.zeros((f_len - ts + DELAY - 1,), dtype=float), in_21[:ts - DELAY + 1]])
                in_2_subset = cc([np.zeros((f_len - ts + DELAY - 1,), dtype=float), in_22[:ts - DELAY + 1]])
                out_subset = cc([np.zeros((f_len - ts + DELAY - 1,), dtype=float), out_2[:ts - DELAY + 1]])
            else:
                in_1_subset = in_21[ts - DELAY + 1 - f_len:ts - DELAY + 1]
                in_2_subset = in_22[ts - DELAY + 1 - f_len:ts - DELAY + 1]
                out_subset = out_2[ts - DELAY + 1 - f_len:ts - DELAY + 1]

            out_2[ts] = in_1_subset.dot(f_in_1[::-1]) + in_2_subset.dot(f_in_2[::-1]) + \
                out_subset.dot(f_out[::-1]) + c + c_confound * confound_2[ts - DELAY] + np.random.normal(0, NOISE)

        # munge time-series data so we can pass it to statsmodels glm fitter
        b_short = b[:30, :3]
        tsm = time_series.Munger()
        tsm.delay = DELAY
        tsm.basis_in = [None, b_short, b_short]
        tsm.basis_out = b_short
        tsm.orthogonalize_basis()
        feature_matrix_1, response_vector_1 = tsm.munge([confound_1, in_11, in_12], out_1, start=f_len)

        # make sure the shapes are correct (for good luck!)
        self.assertEqual(feature_matrix_1.shape[0], T - f_len - DELAY)
        self.assertEqual(feature_matrix_1.shape[1], 11)
        self.assertEqual(len(response_vector_1), T - f_len - DELAY)

        # fit an ordinary least squares model with statsmodels
        link_function = sm.genmod.families.links.identity
        family = sm.families.Gaussian(link=link_function)
        model = sm.GLM(endog=response_vector_1, exog=feature_matrix_1, family=family)
        results = model.fit()
        prediction_1 = model.predict(results.params, exog=feature_matrix_1)
        feature_matrix_2, response_vector_2 = tsm.munge([confound_2, in_21, in_22], out_2, start=f_len)
        prediction_2 = model.predict(results.params, exog=feature_matrix_2)

        # reconstruct filters from coefficients
        constant, in_filters, out_filter = tsm.filters_from_coeffs(results.params)

        # do the same fit except with the full basis function set
        tsm = time_series.Munger()
        tsm.delay = DELAY
        tsm.basis_in = [None, b, b]
        tsm.basis_out = b
        tsm.orthogonalize_basis()
        feature_matrix_1, response_vector_1 = tsm.munge([confound_1, in_11, in_12], out_1, start=f_len)

        # make sure the shapes are correct (for good luck!)
        self.assertEqual(feature_matrix_1.shape[0], T - f_len - DELAY)
        self.assertEqual(feature_matrix_1.shape[1], 20)
        self.assertEqual(len(response_vector_1), T - f_len - DELAY)

        # fit an ordinary least squares model with statsmodels
        link_function = sm.genmod.families.links.identity
        family = sm.families.Gaussian(link=link_function)
        model_full = sm.GLM(endog=response_vector_1, exog=feature_matrix_1, family=family)
        results_full = model_full.fit()

        # reconstruct filters from coefficients
        constant_full, in_filters_full, out_filter_full = tsm.filters_from_coeffs(results_full.params)

        feature_matrix_2_full, response_vector_2_full = tsm.munge([confound_2, in_21, in_22], out_2, start=f_len)
        prediction_2_full = model_full.predict(results_full.params, exog=feature_matrix_2_full)

        self.assertEqual(len(prediction_2), len(prediction_2_full))

        resid_test = (prediction_2 - response_vector_2)**2
        resid_test_full = (prediction_2_full - response_vector_2_full)**2

        print('True constant: {}'.format(c))
        print('Recovered constant: {}'.format(constant))
        print('Recovered constant (full): {}'.format(constant_full))
        print('Test residual: {}'.format(resid_test.sum()))
        print('Test residual (full model): {}'.format(resid_test_full.sum()))

        # make figure to output test results
        fig = plt.figure(figsize=(10, 8), tight_layout=True)
        ax_ts_0 = fig.add_subplot(4, 1, 1)
        ax_ts_1 = fig.add_subplot(4, 1, 3)
        ax_resid = fig.add_subplot(4, 1, 4)
        ax_filt_0 = fig.add_subplot(4, 2, 3)
        ax_filt_1 = fig.add_subplot(4, 2, 4, sharey=ax_filt_0)
        ax_filts = [ax_filt_0, ax_filt_1]

        ax_ts_0.plot(np.transpose([in_11, in_12, out_1]))
        ax_ts_0.plot(np.arange(T)[-len(prediction_1):], prediction_1, 'r--')
        ax_ts_0.set_xlabel('t')
        ax_ts_0.set_ylabel('input and output')
        ax_ts_0.set_title('training time series')

        ax_ts_1.plot(np.transpose([in_21, in_22, out_2]))
        ax_ts_1.plot(np.arange(T)[-len(prediction_2):], prediction_2, 'r--')
        ax_ts_1.set_xlabel('t')
        ax_ts_1.set_ylabel('input and output')
        ax_ts_1.set_title('test time series')

        ax_resid.plot(np.arange(T)[-len(prediction_2):], resid_test, 'k--')
        ax_resid.plot(np.arange(T)[-len(prediction_2):], resid_test_full, 'k')

        ax_resid.set_xlabel('t')
        ax_resid.set_ylabel('residual')

        ax_filts[0].plot(t, f_in_1, 'b')
        ax_filts[0].plot(t, f_in_2, 'g')
        ax_filts[0].plot(t, f_out, 'r')
        ax_filts[0].set_xlim(t[0], t[-1])
        ax_filts[0].set_xlabel('t')
        ax_filts[0].set_ylabel('filter strength')
        ax_filts[0].set_title('true filters')

        ax_filts[1].plot(t[:30], in_filters[1], 'b--')
        ax_filts[1].plot(t[:30], in_filters[2], 'g--')
        ax_filts[1].plot(t[:30], out_filter, 'r--')
        ax_filts[1].plot(t, in_filters_full[1], 'b')
        ax_filts[1].plot(t, in_filters_full[2], 'g')
        ax_filts[1].plot(t, out_filter_full, 'r')
        ax_filts[1].set_xlim(t[0], t[-1])
        ax_filts[1].set_xlabel('t')
        ax_filts[1].set_ylabel('filter strength')
        ax_filts[1].set_title('recovered filters')

        SAVE_PATH = os.path.join(
            FIGURE_SAVE_DIR,
            'test_prediction_error_decreases_when_filters_are_allowed_to_take_on_proper_timescale.png'
        )
        fig.savefig(SAVE_PATH)

        print('Figure saved at {}'.format(SAVE_PATH))


if __name__ == '__main__':
    unittest.main()