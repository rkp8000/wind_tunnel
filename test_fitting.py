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

import fitting

FIGURE_SAVE_DIR = '/Users/rkp/Desktop'


class GLMFitterTestCase(unittest.TestCase):

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

        fitter = fitting.GLMFitter(link='identity', family='Gaussian')
        fitter.set_params(delay=DELAY, basis_in=[b, b], basis_out=b)

        data = [([in_1, in_2], out)]
        fitter.fit(data=data, start=f_len)
        constant, in_filters, out_filter = fitter.constant, fitter.in_filters, fitter.out_filter
        prediction = fitter.predict()

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

    def test_recovery_of_filters_from_basic_MA_fit(self):
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

        # create filtered output stimulus
        out = np.zeros((T,), dtype=float)
        f_len = len(f_in_1)  # filter length

        # go through each timestep from delay to end (zero padding early ones) and compute output
        # based on filtered input and filtered history
        for ts in range(DELAY, T):
            if ts < DELAY + f_len:
                in_1_subset = cc([np.zeros((f_len - ts + DELAY - 1,), dtype=float), in_1[:ts - DELAY + 1]])
                in_2_subset = cc([np.zeros((f_len - ts + DELAY - 1,), dtype=float), in_2[:ts - DELAY + 1]])
            else:
                in_1_subset = in_1[ts - DELAY + 1 - f_len:ts - DELAY + 1]
                in_2_subset = in_2[ts - DELAY + 1 - f_len:ts - DELAY + 1]

            out[ts] = in_1_subset.dot(f_in_1[::-1]) + in_2_subset.dot(f_in_2[::-1]) + c
            out[ts] += np.random.normal(0, NOISE)

        fitter = fitting.GLMFitter(link='identity', family='Gaussian')
        fitter.set_params(delay=DELAY, basis_in=[b, b], basis_out=False)

        data = [([in_1, in_2], out)]
        fitter.fit(data=data, start=f_len)
        constant, in_filters, out_filter = fitter.constant, fitter.in_filters, fitter.out_filter
        prediction = fitter.predict()

        self.assertFalse(out_filter)
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
        ax_filts[0].set_xlim(t[0], t[-1])
        ax_filts[0].set_xlabel('t')
        ax_filts[0].set_ylabel('filter strength')
        ax_filts[0].set_title('true filters')

        ax_filts[1].plot(t, in_filters[0], 'b')
        ax_filts[1].plot(t, in_filters[1], 'g')
        ax_filts[1].set_xlim(t[0], t[-1])
        ax_filts[1].set_xlabel('t')
        ax_filts[1].set_ylabel('filter strength')
        ax_filts[1].set_title('recovered filters')

        SAVE_PATH = os.path.join(
            FIGURE_SAVE_DIR,
            'test_recovery_of_filters_from_basic_MA_fit.png'
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
        N_TRAIN = 5
        NOISE = 0.2
        SCALING = 0.2
        DELAY = 5
        SMOOTH = 3
        cc = np.concatenate

        # make stimuli
        confound_train = [ndimage.gaussian_filter1d(np.random.normal(0, 1, (T,)), SMOOTH) for _ in range(N_TRAIN)]
        in_1_train = [ndimage.gaussian_filter1d(np.random.normal(0, 1, (T,)), SMOOTH) for _ in range(N_TRAIN)]
        in_2_train = [ndimage.gaussian_filter1d(np.random.normal(0, 2, (T,)), SMOOTH) for _ in range(N_TRAIN)]

        confound_test = [ndimage.gaussian_filter1d(np.random.normal(0, 1, (T,)), SMOOTH)]
        in_1_test = [ndimage.gaussian_filter1d(np.random.normal(0, 1, (T,)), SMOOTH)]
        in_2_test = [ndimage.gaussian_filter1d(np.random.normal(0, 2, (T,)), SMOOTH)]

        c = -0.4
        c_confound = 0.2
        t = np.arange(0, 100)
        b_1 = np.exp(-t/5)/5
        b_2 = np.exp(-t/10)/10
        b_3 = np.exp(-t/15)/15
        b_4 = np.exp(-t/50)/50
        b_5 = np.exp(-t/60)/60
        b_6 = np.exp(-t/70)/70

        # full basis (used to generate ground truth output)
        b = np.array([b_1, b_2, b_3, b_4, b_5, b_6]).T

        f_in_1 = b.dot(SCALING*np.array([1, 1, 3, 0, 20, 0])[:, None])
        f_in_2 = b.dot(SCALING*np.array([-1, 3, -1, 0, 20, 0])[:, None])
        f_out = b.dot(SCALING*np.array([.5, 4, -.1, 0, 0, 0])[:, None])

        # create filtered output stimulus
        out_train = [np.zeros((T,), dtype=float) for _ in range(N_TRAIN)]
        out_test = [np.zeros((T,), dtype=float)]
        f_len = len(f_out)  # filter length

        # go through each timestep from delay to end (zero padding early ones) and compute output
        # based on filtered input and filtered history

        # training signals
        data_train = []
        for confound, in_1, in_2, out in zip(confound_train, in_1_train, in_2_train, out_train):
            for ts in range(DELAY, T):
                if ts < DELAY + f_len:
                    in_1_subset = cc([np.zeros((f_len - ts + DELAY - 1,), dtype=float), in_1[:ts - DELAY + 1]])
                    in_2_subset = cc([np.zeros((f_len - ts + DELAY - 1,), dtype=float), in_2[:ts - DELAY + 1]])
                    out_subset = cc([np.zeros((f_len - ts + DELAY - 1,), dtype=float), out[:ts - DELAY + 1]])
                else:
                    in_1_subset = in_1[ts - DELAY + 1 - f_len:ts - DELAY + 1]
                    in_2_subset = in_2[ts - DELAY + 1 - f_len:ts - DELAY + 1]
                    out_subset = out[ts - DELAY + 1 - f_len:ts - DELAY + 1]

                out[ts] = in_1_subset.dot(f_in_1[::-1]) + in_2_subset.dot(f_in_2[::-1]) + \
                    out_subset.dot(f_out[::-1]) + c + c_confound * confound[ts - DELAY] + np.random.normal(0, NOISE)
            data_train.append(([confound, in_1, in_2], out))

        # testing signals
        data_test = []
        for confound, in_1, in_2, out in zip(confound_test, in_1_test, in_2_test, out_test):
            for ts in range(DELAY, T):
                if ts < DELAY + f_len:
                    in_1_subset = cc([np.zeros((f_len - ts + DELAY - 1,), dtype=float), in_1[:ts - DELAY + 1]])
                    in_2_subset = cc([np.zeros((f_len - ts + DELAY - 1,), dtype=float), in_2[:ts - DELAY + 1]])
                    out_subset = cc([np.zeros((f_len - ts + DELAY - 1,), dtype=float), out[:ts - DELAY + 1]])
                else:
                    in_1_subset = in_1[ts - DELAY + 1 - f_len:ts - DELAY + 1]
                    in_2_subset = in_2[ts - DELAY + 1 - f_len:ts - DELAY + 1]
                    out_subset = out[ts - DELAY + 1 - f_len:ts - DELAY + 1]

                out[ts] = in_1_subset.dot(f_in_1[::-1]) + in_2_subset.dot(f_in_2[::-1]) + \
                    out_subset.dot(f_out[::-1]) + c + c_confound * confound[ts - DELAY] + np.random.normal(0, NOISE)
            data_test.append(([confound, in_1, in_2], out))

        # make too-short basis
        b_short = b[:30, :3]

        # make "over-full" basis
        t_of = np.arange(0, 180)
        b_1 = np.exp(-t_of/5)/5
        b_2 = np.exp(-t_of/10)/10
        b_3 = np.exp(-t_of/15)/15
        b_4 = np.exp(-t_of/50)/50
        b_5 = np.exp(-t_of/60)/60
        b_6 = np.exp(-t_of/70)/70
        b_7 = np.exp(-t_of/100)/100
        b_8 = np.exp(-t_of/130)/130

        b_of = np.array([b_1, b_2, b_3, b_4, b_5, b_6, b_7, b_8]).T

        # make basic fitter
        fitter = fitting.GLMFitter(link='identity', family='Gaussian')
        fitter.set_params(delay=DELAY, basis_in=[None, b_short, b_short], basis_out=b_short)
        fitter.fit(data=data_train, start=f_len)

        constant = fitter.constant
        in_filters = fitter.in_filters
        out_filter = fitter.out_filter

        prediction_test = fitter.predict(data_test, start=f_len)
        _, response_vector_test = fitter.make_feature_matrix_and_response_vector(data_test, f_len)

        # make "full" fitter
        fitter_full = fitting.GLMFitter(link='identity', family='Gaussian')
        fitter_full.set_params(delay=DELAY, basis_in=[None, b, b], basis_out=b)
        fitter_full.fit(data=data_train, start=f_len)

        constant_full = fitter_full.constant
        in_filters_full = fitter_full.in_filters
        out_filter_full = fitter_full.out_filter

        prediction_test_full = fitter_full.predict(data_test, start=f_len)
        _, response_vector_test_full = fitter_full.make_feature_matrix_and_response_vector(data_test, f_len)

        # make "overfull" fitter
        fitter_of = fitting.GLMFitter(link='identity', family='Gaussian')
        fitter_of.set_params(delay=DELAY, basis_in=[None, b_of, b_of], basis_out=b_of)

        fitter_of.fit(data=data_train, start=f_len)

        # reconstruct filters from coefficients
        constant_of = fitter_of.constant
        in_filters_of = fitter_of.in_filters
        out_filter_of = fitter_of.out_filter

        prediction_test_of = fitter_of.predict(data_test, start=f_len)
        _, response_vector_test_of = fitter_of.make_feature_matrix_and_response_vector(data_test, f_len)

        self.assertEqual(len(prediction_test), len(prediction_test_full))
        self.assertEqual(len(prediction_test), len(prediction_test_of))

        resid_test = (prediction_test - response_vector_test)**2
        resid_test_full = (prediction_test_full - response_vector_test_full)**2
        resid_test_of = (prediction_test_of - response_vector_test_of)**2

        print('True constant: {}'.format(c))
        print('Recovered constant: {}'.format(constant))
        print('Recovered constant (full): {}'.format(constant_full))
        print('Recovered constant (overfull): {}'.format(constant_of))
        print('Test residual: {}'.format(resid_test.sum()))
        print('Test residual (full model): {}'.format(resid_test_full.sum()))
        print('Test residual (overfull model): {}'.format(resid_test_of.sum()))

        # make figure to output test results
        fig = plt.figure(figsize=(10, 8), tight_layout=True)
        ax_ts_0 = fig.add_subplot(4, 1, 1)
        ax_ts_1 = fig.add_subplot(4, 1, 3)
        ax_resid = fig.add_subplot(4, 1, 4)
        ax_filt_0 = fig.add_subplot(4, 2, 3)
        ax_filt_1 = fig.add_subplot(4, 2, 4, sharey=ax_filt_0)
        ax_filts = [ax_filt_0, ax_filt_1]

        ax_ts_0.plot(np.transpose([in_1_train[0], in_2_train[0], out_train[0]]))
        ax_ts_0.set_xlabel('t')
        ax_ts_0.set_ylabel('input and output')
        ax_ts_0.set_title('training time series')

        ax_ts_1.plot(np.transpose([in_1_test[0], in_2_test[0], out_test[0]]))
        ax_ts_1.plot(np.arange(T)[-len(prediction_test):], prediction_test, 'r--')
        ax_ts_1.set_xlabel('t')
        ax_ts_1.set_ylabel('input and output')
        ax_ts_1.set_title('test time series')

        ax_resid.plot(np.arange(T)[-len(prediction_test):], resid_test, 'k--')
        ax_resid.plot(np.arange(T)[-len(prediction_test):], resid_test_full, 'k')

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