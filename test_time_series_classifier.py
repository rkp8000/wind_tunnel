"""
Unit tests for time-series classifiers.
"""
import matplotlib.pyplot as plt
import numpy as np
import unittest

import time_series_classifier as tsc

cc = np.concatenate


class VARTestCase(unittest.TestCase):

    def setUp(self):
        # specify parameters
        self.a_1 = np.array([
            [[.9, 0, -.1],
             [-.2, .9, .1],
             [-.1, 0, .85]],
            [[.1, 0, -.01],
             [.1, .2, .05],
             [.05, 0, .3]],
        ])
        self.k_1 = np.array([
            [.2, .1, -.05],
            [.1, .3, 0],
            [-.05, 0, .3],
        ])
        self.a_2 = np.array([
            [[.5, -.2, .06],
             [-.1, .2, .1],
             [-.2, 0, .3]],
            [[.6, -.2, .05],
             [.04, .8, -.1],
             [.2, 0.1, .5]],
        ])
        self.k_2 = np.array([
            [.2, .1, -.05],
            [.1, .3, 0],
            [-.05, 0, .3],
        ])

    def test_we_can_classify_time_series_generated_from_ideal_distributions(self):

        var_model_1 = tsc.VarModel(a=self.a_1, k=self.k_1)
        var_model_2 = tsc.VarModel(a=self.a_2, k=self.k_2)

        tss_1 = [var_model_1.sample(initial='zero', t=500) for _ in range(20)]
        tss_2 = [var_model_2.sample(initial='zero', t=500) for _ in range(20)]

        # make sure correct length time-series were generated
        self.assertEqual(len(tss_1[0]), 500)
        self.assertEqual(len(tss_1[-1]), 500)
        self.assertEqual(len(tss_2[0]), 500)
        self.assertEqual(len(tss_2[-1]), 500)

        # make sure beginnings are all zero
        for ts in [tss_1[0], tss_1[-1], tss_2[0], tss_2[-1]]:
            np.testing.assert_array_almost_equal(ts[:2, :3], np.zeros((2, 3)))

        # make sure that probability of time-series are highest under the models that made them
        for _ in range(10):
            ts_1 = np.random.choice(tss_1)
            ts_2 = np.random.choice(tss_2)

            self.assertGreater(var_model_1.log_prob(ts_1), var_model_2.log_prob(ts_1))
            self.assertGreater(var_model_2.log_prob(ts_2), var_model_1.log_prob(ts_2))

        # make sure we can classify time-series correctly
        f = tsc.VarClassifierBinary()
        f.train(positives=tss_1, negatives=tss_2)
        predictions = f.predict(tss_2 + tss_1)
        ground_truth = cc([-np.ones((len(tss_2),), dtype=int),
                           np.ones((len(tss_1),), dtype=int)])
        np.testing.assert_array_equal(predictions, ground_truth)

        # make some plots
        fig, axs = plt.subplots(3, 2)
        for ax_row, ts_1, ts_2 in zip(axs, tss_1, tss_2):
            ax_row[0].plot(ts_1)
            ax_row[1].plot(ts_2)

        axs[0, 0].set_title('model 1')
        axs[0, 1].set_title('model 2')
        plt.show()


if __name__ == '__main__':
    unittest.main()