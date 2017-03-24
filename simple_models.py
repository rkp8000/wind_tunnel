from __future__ import division, print_function
import numpy as np
from sklearn import linear_model


class ThresholdLinearHeadingConcModel(object):

    def __init__(self, include_c_max_coefficient):

        self.include_c_max_coefficient = include_c_max_coefficient
        self.threshold = None

        self.linear_models = {
            'below': linear_model.LinearRegression(),
            'above': linear_model.LinearRegression(),
        }

    def brute_force_fit(self, hs, c_maxs, x_0s, h_0s):

        # try all thresholds that will separate data into unique classes
        c_maxs_sorted = np.array(sorted(np.unique(c_maxs)))
        thresholds = 0.5 * (c_maxs_sorted[:-1] + c_maxs_sorted[1:])

        # group predictors together
        predictors = np.array([c_maxs, x_0s, h_0s]).T
        rss_best = np.inf

        for threshold in thresholds:

            # split data
            above_mask = c_maxs >= threshold

            # fit models and get predictions
            ## when c_max is below putative threshold

            lm_below = linear_model.LinearRegression()
            lm_below.fit(predictors[~above_mask, 1:], hs[~above_mask])
            rss_below = np.sum(
                (hs[~above_mask] - lm_below.predict(predictors[~above_mask, 1:])) ** 2)

            ## when c_max is above putative threshold

            lm_above = linear_model.LinearRegression()

            if self.include_c_max_coefficient:
                lm_above.fit(predictors[above_mask, :], hs[above_mask])
                rss_above = np.sum(
                    (hs[above_mask] - lm_above.predict(predictors[above_mask, :])) ** 2)

            else:
                lm_above.fit(predictors[above_mask, 1:], hs[above_mask])
                rss_above = np.sum(
                    (hs[above_mask] - lm_above.predict(predictors[above_mask, 1:])) ** 2)

            if rss_below + rss_above < rss_best:
                rss_best = rss_below + rss_above

                self.threshold = threshold
                self.linear_models['below'] = lm_below
                self.linear_models['above'] = lm_above

    def predict(self, c_maxs, x_0s, h_0s):

        predictors = np.array([c_maxs, x_0s, h_0s]).T
        above_mask = c_maxs >= self.threshold
        hs_predicted = np.nan * np.zeros(c_maxs.shape)
        hs_predicted[~above_mask] = \
            self.linear_models['below'].predict(predictors[~above_mask, 1:])

        if self.include_c_max_coefficient:

            hs_predicted[above_mask] = \
                self.linear_models['above'].predict(predictors[above_mask])

        else:

            hs_predicted[above_mask] = \
                self.linear_models['above'].predict(predictors[above_mask, 1:])

        return hs_predicted