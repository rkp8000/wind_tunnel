from __future__ import division, print_function
import numpy as np
import unittest


class ThresholdLinearHeadingConcModelTestCase(unittest.TestCase):

    def test_model_fit_works_on_example_datasets(self):

        import simple_models

        # build artificial dataset

        c_maxs = np.random.uniform(0, 3, 20)
        x_0s = np.random.uniform(-3, 3, 20)
        h_0s = np.random.uniform(-3, 3, 20)

        c_max_th = 1.5

        hs = np.nan * np.zeros(c_maxs.shape)

        above_mask = c_maxs > c_max_th

        hs[~above_mask] = x_0s[~above_mask] - 2 * h_0s[~above_mask]
        hs[above_mask] = -x_0s[above_mask] + 2 * h_0s[above_mask] + 3 * c_maxs[above_mask]

        # fit model

        tlchm = simple_models.ThresholdLinearHeadingConcModel(include_c_max_coefficient=True)

        tlchm.brute_force_fit(hs, c_maxs, x_0s, h_0s)

        hs_predicted = tlchm.predict(c_maxs, x_0s, h_0s)

        rss = np.sum((hs_predicted - hs) ** 2)

        np.testing.assert_array_almost_equal(hs_predicted, hs)
        self.assertAlmostEqual(rss, 0)

        # make sure that fit is better when include_c_max_coefficient is False

        tlchm2 = simple_models.ThresholdLinearHeadingConcModel(include_c_max_coefficient=False)

        tlchm2.brute_force_fit(hs, c_maxs, x_0s, h_0s)

        hs_predicted2 = tlchm2.predict(c_maxs, x_0s, h_0s)

        rss2 = np.sum((hs_predicted2 - hs) ** 2)

        self.assertGreater(rss2, 0)


if __name__ == '__main__':

    unittest.main()