"""
Unit tests for time-series module.
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


class SegmentingTestCase(unittest.TestCase):

    def test_segment_basic_segments_correctly(self):
        """
        A few example time-serieses to be segmented.
        """

        ts = np.array([False, False, True, True, False, False, True, False])
        starts_correct, ends_correct = (np.array([2, 6]), np.array([4, 7]))
        starts, ends = time_series.segment_basic(ts)

        np.testing.assert_array_equal(starts, starts_correct)
        np.testing.assert_array_equal(ends, ends_correct)

        ts = np.array([False, False, True, True, False, False, True, True])
        starts_correct, ends_correct = (np.array([2, 6]), np.array([4, 8]))
        starts, ends = time_series.segment_basic(ts)

        np.testing.assert_array_equal(starts, starts_correct)
        np.testing.assert_array_equal(ends, ends_correct)

        ts = np.array([True, False, True, True, False, False, True, True])
        starts_correct, ends_correct = (np.array([0, 2, 6]), np.array([1, 4, 8]))
        starts, ends = time_series.segment_basic(ts)

        np.testing.assert_array_equal(starts, starts_correct)
        np.testing.assert_array_equal(ends, ends_correct)

    def test_segment_basic_segments_correctly_with_external_idxs(self):

        t = np.arange(100, 200)

        ts = np.array([False, False, True, True, False, False, True, False])
        starts_correct, ends_correct = (np.array([2, 6]) + 100, np.array([4, 7]) + 100)
        starts, ends = time_series.segment_basic(ts, t)

        np.testing.assert_array_equal(starts, starts_correct)
        np.testing.assert_array_equal(ends, ends_correct)

        ts = np.array([False, False, True, True, False, False, True, True])
        starts_correct, ends_correct = (np.array([2, 6]) + 100, np.array([4, 8]) + 100)
        starts, ends = time_series.segment_basic(ts, t)

        np.testing.assert_array_equal(starts, starts_correct)
        np.testing.assert_array_equal(ends, ends_correct)

        ts = np.array([True, False, True, True, False, False, True, True])
        starts_correct, ends_correct = (np.array([0, 2, 6]) + 100, np.array([1, 4, 8]) + 100)
        starts, ends = time_series.segment_basic(ts, t)

        np.testing.assert_array_equal(starts, starts_correct)
        np.testing.assert_array_equal(ends, ends_correct)

    def test_segment_by_threshold_gives_correct_answer_for_examples(self):

        threshold = 5

        t = np.arange(100, 200)
        t_extended = np.arange(100, 201)

        # signal with two threshold crossings
        x = np.random.uniform(0, 2, t.shape)
        x[10:20] = 10 + np.random.uniform(0, 2, (10,))  # above threshold
        x[60:75] = 20 + np.random.uniform(0, 2, (15,))  # above threshold

        x[15] = 15  # peak 1
        x[70] = 25  # peak 2

        # correct solution
        segments_correct = np.array([[100, 110, 115, 120, 160],
                                     [120, 160, 170, 175, 200]])
        peaks_correct = np.array([15., 25])

        segments, peaks = time_series.segment_by_threshold(x, threshold, t=t_extended)

        np.testing.assert_array_equal(segments, segments_correct)
        np.testing.assert_array_equal(peaks, peaks_correct)

        # signal ending above threshold
        x = np.random.uniform(0, 2, t.shape)
        x[10:20] = 10 + np.random.uniform(0, 2, (10,))  # above threshold
        x[60:] = 20 + np.random.uniform(0, 2, (40,))  # above threshold

        x[15] = 15  # peak 1
        x[70] = 25  # peak 2

        # correct solution
        segments_correct = np.array([[100, 110, 115, 120, 160],
                                     [120, 160, 170, 200, 200]])
        peaks_correct = np.array([15., 25])

        segments, peaks = time_series.segment_by_threshold(x, threshold, t=t_extended)

        np.testing.assert_array_equal(segments, segments_correct)
        np.testing.assert_array_equal(peaks, peaks_correct)

        # signal starting above threshold
        x = np.random.uniform(0, 2, t.shape)
        x[:20] = 10 + np.random.uniform(0, 2, (20,))  # above threshold
        x[60:75] = 20 + np.random.uniform(0, 2, (15,))  # above threshold

        x[15] = 15  # peak 1
        x[70] = 25  # peak 2

        # correct solution
        segments_correct = np.array([[100, 100, 115, 120, 160],
                                     [120, 160, 170, 175, 200]])
        peaks_correct = np.array([15., 25])

        segments, peaks = time_series.segment_by_threshold(x, threshold, t=t_extended)

        np.testing.assert_array_equal(segments, segments_correct)
        np.testing.assert_array_equal(peaks, peaks_correct)

    def test_segment_by_threshold_gives_correct_answer_for_one_or_zero_threshold_crossings(self):

        threshold = 5

        # no crossings
        x = np.random.uniform(0, 1, 100)

        segments, peaks = time_series.segment_by_threshold(x, threshold)

        self.assertEqual(segments.shape, (0, 5))
        self.assertEqual(len(peaks), 0)

        # one crossing
        x[50:60] = 10
        x[55] = 15

        segments_correct = np.array([[0, 50, 55, 60, 100]])
        peaks_correct = np.array([15])

        segments, peaks = time_series.segment_by_threshold(x, threshold)

        np.testing.assert_array_equal(segments, segments_correct)
        np.testing.assert_array_equal(peaks, peaks_correct)


class CrossCovarianceTestCase(unittest.TestCase):

    def setUp(self):
        print("In test '{}'...".format(self._testMethodName))

    def test_autocovariance_is_delta_function_for_white_noise(self):

        xs = [np.random.normal(0, 3, np.random.randint(500, 1000)) for _ in range(100)]

        acov, pv, conf_lb, conf_ub = \
            time_series.xcov_multi_with_confidence(xs, xs, lag_backward=20, lag_forward=21, normed=True)

        self.assertEqual(len(acov), 41)
        self.assertAlmostEqual(acov[:20].mean(), 0, places=2)
        self.assertAlmostEqual(acov[21:].mean(), 0, places=2)
        self.assertAlmostEqual(acov[20], 1, places=5)

        np.testing.assert_array_almost_equal(acov[:20], acov[21:][::-1])
        np.testing.assert_array_almost_equal(pv[:20], pv[21:][::-1])
        np.testing.assert_array_almost_equal(conf_lb[:20], conf_lb[21:][::-1])
        np.testing.assert_array_almost_equal(conf_ub[:20], conf_ub[21:][::-1])

    def test_autocovariance_is_extended_for_smoother_signals_and_has_correct_confidences(self):

        xs = [np.random.normal(0, 1, np.random.randint(500, 1000)) for _ in range(100)]
        xs_smooth = [ndimage.gaussian_filter1d(x, 5) for x in xs]

        acov, _, lb, ub = time_series.\
            xcov_multi_with_confidence(xs, xs, lag_backward=20, lag_forward=21, normed=True)
        acov_smooth, _, lb_smooth, ub_smooth = time_series.\
            xcov_multi_with_confidence(xs_smooth, xs_smooth, lag_backward=20, lag_forward=21, normed=True)

        for lag in range(2, 21):
            self.assertLess(acov[20 + lag], acov_smooth[20 + lag])
            self.assertLess(acov[20 + lag], acov_smooth[20 + lag])
            self.assertLess(acov[20 - lag], acov_smooth[20 - lag])
            self.assertLess(acov[20 - lag], acov_smooth[20 - lag])

            self.assertGreater(acov_smooth[20 + lag], lb_smooth[20 + lag])
            self.assertLess(acov_smooth[20 + lag], ub_smooth[20 + lag])
            self.assertGreater(acov_smooth[20 - lag], lb_smooth[20 - lag])
            self.assertLess(acov_smooth[20 - lag], ub_smooth[20 - lag])

        np.testing.assert_array_less(lb[:20], ub[:20])
        self.assertTrue(np.all(lb_smooth[:20] > -1))
        self.assertTrue(np.all(ub_smooth[:20] < 1))

    def test_unnormed_autocovariance_gives_variance_at_0_lag(self):

        xs = [np.random.normal(0, 1, np.random.randint(500, 1000)) for _ in range(100)]
        xs = [ndimage.gaussian_filter1d(x, 5) for x in xs]

        acov, _, lb, ub = time_series.\
            xcov_multi_with_confidence(xs, xs, lag_backward=20, lag_forward=21, normed=False)

        self.assertAlmostEqual(acov[20], np.var(np.concatenate(xs)), places=5)

        for lag in range(2, 21):
            self.assertGreater(acov[20 + lag], lb[20 + lag])
            self.assertLess(acov[20 + lag], ub[20 + lag])
            self.assertGreater(acov[20 - lag], lb[20 - lag])
            self.assertLess(acov[20 - lag], ub[20 - lag])


class TimeSeriesMungerTestCase(unittest.TestCase):

    def test_feature_matrix_correctly_formed_with_basis_functions(self):
        # three features, 1 output
        x1 = np.random.normal(0, 10, (100,))
        x2 = np.random.normal(0, 10, (100,))
        x3 = np.random.normal(0, 10, (100,))
        y = np.random.normal(0, 10, (100,))

        tsm = time_series.Munger()
        tsm.delay = 10  # we will assume there is a 10 timestep delay between input and response

        # we will assume x1 is used directly and x2, x3, and y are filtered, and that their
        # filters are represented by sums of sinusoidal basis functions
        # (the details don't matter as we're just making sure all the arrays are getting
        # rearranged correctly)
        t = np.linspace(0, np.pi, 20)
        sin_basis = np.transpose([np.sin(t), np.sin(2*t), np.sin(3*t), np.sin(4*t)])
        t_short = t[:10]
        sin_basis_short = np.transpose([np.sin(t_short), np.sin(2*t_short)])
        tsm.basis_in = [None, sin_basis, sin_basis]
        tsm.basis_out = sin_basis_short

        feature_matrix, response_vector = tsm.munge([x1, x2, x3], y)

        # make sure the arrays are the correct shape
        self.assertEqual(len(response_vector), 90)
        self.assertEqual(feature_matrix.shape[0], 90)
        self.assertEqual(feature_matrix.shape[1], 12)

        # make sure the constants and x1 terms are in the correct places (check 1st, 2nd, last)
        self.assertAlmostEqual(feature_matrix[0, 0], 1)
        self.assertAlmostEqual(feature_matrix[1, 0], 1)
        self.assertAlmostEqual(feature_matrix[-1, 0], 1)
        self.assertAlmostEqual(feature_matrix[0, 1], x1[0])
        self.assertAlmostEqual(feature_matrix[1, 1], x1[1])
        self.assertAlmostEqual(feature_matrix[-1, 1], x1[89])

        # make sure the rest of the features correspond to the projections of x2, x3, and y
        # onto their appropriate basis functions
        np.testing.assert_array_almost_equal(
            feature_matrix[19, 2:6],
            x2[:20][None, :].dot(sin_basis[::-1]).flatten()
        )
        np.testing.assert_array_almost_equal(
            feature_matrix[19, 6:10],
            x3[:20][None, :].dot(sin_basis[::-1]).flatten()
        )
        np.testing.assert_array_almost_equal(
            feature_matrix[9, 10:12],
            y[:10][None, :].dot(sin_basis_short[::-1]).flatten()
        )

        np.testing.assert_array_almost_equal(
            feature_matrix[20, 2:6],
            x2[1:21][None, :].dot(sin_basis[::-1]).flatten()
        )
        np.testing.assert_array_almost_equal(
            feature_matrix[20, 6:10],
            x3[1:21][None, :].dot(sin_basis[::-1]).flatten()
        )
        np.testing.assert_array_almost_equal(
            feature_matrix[10, 10:12],
            y[1:11][None, :].dot(sin_basis_short[::-1]).flatten()
        )

        np.testing.assert_array_almost_equal(
            feature_matrix[-1, 2:6],
            x2[70:90][None, :].dot(sin_basis[::-1]).flatten()
        )
        np.testing.assert_array_almost_equal(
            feature_matrix[-1, 6:10],
            x3[70:90][None, :].dot(sin_basis[::-1]).flatten()
        )
        np.testing.assert_array_almost_equal(
            feature_matrix[-1, 10:12],
            y[80:90][None, :].dot(sin_basis_short[::-1]).flatten()
        )

    def test_feature_matrix_correctly_formed_with_basis_functions_with_nonzero_start(self):
        # three features, 1 output
        x1 = np.random.normal(0, 10, (100,))
        x2 = np.random.normal(0, 10, (100,))
        x3 = np.random.normal(0, 10, (100,))
        y = np.random.normal(0, 10, (100,))

        tsm = time_series.Munger()
        tsm.delay = 10  # we will assume there is a 10 timestep delay between input and response

        # we will assume x1 is used directly and x2, x3, and y are filtered, and that their
        # filters are represented by sums of sinusoidal basis functions
        # (the details don't matter as we're just making sure all the arrays are getting
        # rearranged correctly)
        t = np.linspace(0, np.pi, 20)
        sin_basis = np.transpose([np.sin(t), np.sin(2*t), np.sin(3*t), np.sin(4*t)])
        t_short = t[:10]
        sin_basis_short = np.transpose([np.sin(t_short), np.sin(2*t_short)])
        tsm.basis_in = [None, sin_basis, sin_basis]
        tsm.basis_out = sin_basis_short

        feature_matrix, response_vector = tsm.munge([x1, x2, x3], y, start=5)

        # make sure the arrays are the correct shape
        self.assertEqual(len(response_vector), 85)
        self.assertEqual(feature_matrix.shape[0], 85)
        self.assertEqual(feature_matrix.shape[1], 12)

        # make sure the constants and x1 terms are in the correct places (check 1st, 2nd, last)
        self.assertAlmostEqual(feature_matrix[0, 0], 1)
        self.assertAlmostEqual(feature_matrix[1, 0], 1)
        self.assertAlmostEqual(feature_matrix[-1, 0], 1)
        self.assertAlmostEqual(feature_matrix[0, 1], x1[5])
        self.assertAlmostEqual(feature_matrix[1, 1], x1[6])
        self.assertAlmostEqual(feature_matrix[-1, 1], x1[89])

        # make sure the rest of the features correspond to the projections of x2, x3, and y
        # onto their appropriate basis functions
        np.testing.assert_array_almost_equal(
            feature_matrix[14, 2:6],
            x2[:20][None, :].dot(sin_basis[::-1]).flatten()
        )
        np.testing.assert_array_almost_equal(
            feature_matrix[14, 6:10],
            x3[:20][None, :].dot(sin_basis[::-1]).flatten()
        )
        np.testing.assert_array_almost_equal(
            feature_matrix[4, 10:12],
            y[:10][None, :].dot(sin_basis_short[::-1]).flatten()
        )

        np.testing.assert_array_almost_equal(
            feature_matrix[15, 2:6],
            x2[1:21][None, :].dot(sin_basis[::-1]).flatten()
        )
        np.testing.assert_array_almost_equal(
            feature_matrix[15, 6:10],
            x3[1:21][None, :].dot(sin_basis[::-1]).flatten()
        )
        np.testing.assert_array_almost_equal(
            feature_matrix[5, 10:12],
            y[1:11][None, :].dot(sin_basis_short[::-1]).flatten()
        )

        np.testing.assert_array_almost_equal(
            feature_matrix[-1, 2:6],
            x2[70:90][None, :].dot(sin_basis[::-1]).flatten()
        )
        np.testing.assert_array_almost_equal(
            feature_matrix[-1, 6:10],
            x3[70:90][None, :].dot(sin_basis[::-1]).flatten()
        )
        np.testing.assert_array_almost_equal(
            feature_matrix[-1, 10:12],
            y[80:90][None, :].dot(sin_basis_short[::-1]).flatten()
        )

    def test_orthogonalization_of_nonorthogonal_basis_functions(self):

        non_orth_basis_1 = np.array([[0., 1], [1, 1], [0, 0]])
        non_orth_basis_2 = np.random.normal(0, 1, (20, 3))
        non_orth_basis_3 = np.random.uniform(0, 1, (20, 5))

        tsm = time_series.Munger()
        tsm.delay = 4
        tsm.basis_in = [None, non_orth_basis_1, non_orth_basis_2]
        tsm.basis_out = non_orth_basis_3

        tsm.orthogonalize_basis()

        self.assertTrue(tsm.basis_in[0] is None)

        # check orthogonality of basis_in[1]
        self.assertAlmostEqual(tsm.basis_in[1][:, 0].dot(tsm.basis_in[1][:, 1]), 0)
        # check orthogonality of basis_in[2]
        self.assertAlmostEqual(tsm.basis_in[2][:, 0].dot(tsm.basis_in[2][:, 1]), 0)
        self.assertAlmostEqual(tsm.basis_in[2][:, 0].dot(tsm.basis_in[2][:, 2]), 0)
        self.assertAlmostEqual(tsm.basis_in[2][:, 1].dot(tsm.basis_in[2][:, 2]), 0)
        # check orthogonality of basis_out
        self.assertAlmostEqual(tsm.basis_out[:, 0].dot(tsm.basis_out[:, 1]), 0)
        self.assertAlmostEqual(tsm.basis_out[:, 0].dot(tsm.basis_out[:, 2]), 0)
        self.assertAlmostEqual(tsm.basis_out[:, 0].dot(tsm.basis_out[:, 3]), 0)
        self.assertAlmostEqual(tsm.basis_out[:, 0].dot(tsm.basis_out[:, 4]), 0)
        self.assertAlmostEqual(tsm.basis_out[:, 1].dot(tsm.basis_out[:, 2]), 0)
        self.assertAlmostEqual(tsm.basis_out[:, 1].dot(tsm.basis_out[:, 3]), 0)
        self.assertAlmostEqual(tsm.basis_out[:, 1].dot(tsm.basis_out[:, 4]), 0)
        self.assertAlmostEqual(tsm.basis_out[:, 2].dot(tsm.basis_out[:, 3]), 0)
        self.assertAlmostEqual(tsm.basis_out[:, 2].dot(tsm.basis_out[:, 4]), 0)
        self.assertAlmostEqual(tsm.basis_out[:, 3].dot(tsm.basis_out[:, 4]), 0)

    def test_putting_filters_back_together_from_fitted_basis_function_coefficients(self):
        """
        Make sure we can properly put filters back together if we have found their coefficients.
        """
        cc = np.concatenate

        tsm = time_series.Munger()
        tsm.delay = 10  # we will assume there is a 10 timestep delay between input and response

        # we will assume x1 is used directly and x2, x3, and y are filtered, and that their
        # filters are represented by sums of sinusoidal basis functions
        # (the details don't matter as we're just making sure all the arrays are getting
        # rearranged correctly)
        t = np.linspace(0, np.pi, 20)
        sin_basis = np.transpose([np.sin(t), np.sin(2*t), np.sin(3*t), np.sin(4*t)])
        t_short = t[:10]
        sin_basis_short = np.transpose([np.sin(t_short), np.sin(2*t_short)])
        tsm.basis_in = [None, sin_basis, sin_basis]
        tsm.basis_out = sin_basis_short

        coeffs = cc([[-1.], [.3], [1, 2, 3, 4], [5, 6, 7, 8], [-3, -4]])
        constant, in_filters, out_filter = tsm.filters_from_coeffs(coeffs)

        self.assertAlmostEqual(constant, coeffs[0])
        self.assertAlmostEqual(in_filters[0], coeffs[1])
        np.testing.assert_array_almost_equal(in_filters[1], sin_basis.dot(coeffs[2:6]))
        np.testing.assert_array_almost_equal(in_filters[2], sin_basis.dot(coeffs[6:10]))
        np.testing.assert_array_almost_equal(out_filter, sin_basis_short.dot(coeffs[10:12]))

    def test_actual_fitting_of_filters_using_munger(self):
        """
        Create an output time-series by filtering and combining two input time-series and make sure that
        statsmodels can properly recover the filters given the munged data.
        """
        T = 500
        NOISE = 0.01
        SCALING = 0.01
        DELAY = 5
        cc = np.concatenate

        in_1 = ndimage.gaussian_filter1d(np.random.normal(0, 1, (T,)), 1)
        in_2 = ndimage.gaussian_filter1d(np.random.normal(0, 2, (T,)), 3)

        c = -0.4
        t = np.linspace(0, np.pi, 30)
        b_1 = np.sin(t)
        b_2 = np.sin(2*t)
        b_3 = np.sin(3*t)

        b = np.array([b_1, b_2, b_3]).T

        f_in_1 = b.dot(SCALING*np.array([1, 1, 3])[:, None])
        f_in_2 = b.dot(SCALING*np.array([-1, 3, -1])[:, None])
        f_out = b.dot(SCALING*np.array([.5, 4, -.1])[:, None])

        # create filtered output stimulus
        out = np.zeros((T,), dtype=float)
        L = len(f_out)  # filter length

        # go through each timestep from delay to end (zero padding early ones) and compute output
        # based on filtered input and filtered history
        for ts in range(DELAY, T):
            if ts < DELAY + L:
                in_1_subset = cc([np.zeros((L - ts + DELAY - 1,), dtype=float), in_1[:ts - DELAY + 1]])
                in_2_subset = cc([np.zeros((L - ts + DELAY - 1,), dtype=float), in_2[:ts - DELAY + 1]])
                out_subset = cc([np.zeros((L - ts + DELAY - 1,), dtype=float), out[:ts - DELAY + 1]])
            else:
                in_1_subset = in_1[ts - DELAY + 1 - L:ts - DELAY + 1]
                in_2_subset = in_2[ts - DELAY + 1 - L:ts - DELAY + 1]
                out_subset = out[ts - DELAY + 1 - L:ts - DELAY + 1]

            out[ts] = in_1_subset.dot(f_in_1[::-1]) + in_2_subset.dot(f_in_2[::-1]) + out_subset.dot(f_out[::-1]) + c
            out[ts] += np.random.normal(0, NOISE)

        # munge time-series data so we can pass it to statsmodels glm fitter
        tsm = time_series.Munger()
        tsm.delay = DELAY
        tsm.basis_in = [b, b]
        tsm.basis_out = b
        tsm.orthogonalize_basis()
        feature_matrix, response_vector = tsm.munge([in_1, in_2], out, start=L)

        # make sure the shapes are correct (for good luck!)
        self.assertEqual(feature_matrix.shape[0], T - L - DELAY)
        self.assertEqual(feature_matrix.shape[1], 10)
        self.assertEqual(len(response_vector), T - L - DELAY)

        # fit an ordinary least squares model with statsmodels
        link_function = sm.genmod.families.links.identity
        family = sm.families.Gaussian(link=link_function)
        model = sm.GLM(endog=response_vector, exog=feature_matrix, family=family)
        results = model.fit()

        # reconstruct filters from coefficients
        constant, in_filters, out_filter = tsm.filters_from_coeffs(results.params)

        print('True constant: {}'.format(c))
        print('Recovered constant: {}'.format(constant))

        # make figure to output test results
        fig = plt.figure(figsize=(10, 5), tight_layout=True)
        ax_ts = fig.add_subplot(2, 1, 1)
        ax_filts = [fig.add_subplot(2, 2, 3), fig.add_subplot(2, 2, 4)]

        ax_ts.plot(np.transpose([in_1, in_2, out]))
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

        SAVE_PATH = os.path.join(FIGURE_SAVE_DIR, 'test_actual_fitting_of_filters_using_munger.png')
        fig.savefig(os.path.join(SAVE_PATH))

        print('Figure saved at {}'.format(SAVE_PATH))


if __name__ == '__main__':
    unittest.main()