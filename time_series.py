"""
Functions for playing with time-serieses.
"""
from __future__ import print_function, division
import numpy as np
from numpy import concatenate as cc
import stats


def segment_basic(x, t=None):
    """
    Return the numerical indices indicating the segments of non-False x-values.
    :param x: boolean time-series
    :param t: vector containing indices to use if not 0 to len(x)
    :return: starts, ends, which are numpy arrays containing the start and end idxs of segments of consecutive Trues
    """

    if t is None:
        t = np.arange(len(x), dtype=int)

    starts = t[(np.diff(cc([[False], x]).astype(int)) == 1).nonzero()[0]]
    ends = t[(np.diff(cc([x, [False]]).astype(int)) == -1).nonzero()[0]] + 1

    return starts, ends


def xcov_multi_with_confidence(xs, ys, lag_backward, lag_forward, confidence=0.95, normed=False):
    """
    Calculate cross-covariance between x and y when multiple time-series are available.
        This function is to be used when it is believed that y is created by filtering x
        with a causal linear filter. If that is the case, then as the number of samples
        increases, the result will approach the shape of the original filter.
    :param xs: list of input time-series
    :param ys: list of output time-series
    :param lag_forward: number of lags to look forward (causal (x yields y))
    :param lag_backward: number of lags to look back (acausal (y yields x))
    :param confidence: confidence of confidence interval desired
    :param normed: if True, results will be normalized by geometric mean of x's & y's variances
    :return: cross-covariance, p-value, lower bound, upper bound
    """

    if not np.all([len(x) == len(y) for x, y in zip(xs, ys)]):
        raise ValueError('Arrays within xs and ys must all be of the same size!')

    covs = []
    p_values = []
    lbs = []
    ubs = []

    for lag in range(-lag_backward, lag_forward):
        if lag == 0:
            x_rel = xs
            y_rel = ys
        elif lag < 0:
            x_rel = [x[-lag:] for x in xs if len(x) > -lag]
            y_rel = [y[:lag] for y in ys if len(y) > -lag]
        else:
            # calculate the cross covariance between x and y with a specific lag
            # first get the relevant xs and ys from each time-series
            x_rel = [x[:-lag] for x in xs if len(x) > lag]
            y_rel = [y[lag:] for y in ys if len(y) > lag]

        all_xs = np.concatenate(x_rel)
        all_ys = np.concatenate(y_rel)

        cov, p_value, lb, ub = stats.cov_with_confidence(all_xs, all_ys, confidence)

        covs.append(cov)
        p_values.append(p_value)
        lbs.append(lb)
        ubs.append(ub)

    covs = np.array(covs)
    p_values = np.array(p_values)
    lbs = np.array(lbs)
    ubs = np.array(ubs)

    if normed:
        # normalize by average variance of signals
        var_x = np.cov(np.concatenate(xs), np.concatenate(xs))[0, 0]
        var_y = np.cov(np.concatenate(ys), np.concatenate(ys))[0, 0]
        norm_factor = np.sqrt(var_x * var_y)
        covs /= norm_factor
        lbs /= norm_factor
        ubs /= norm_factor

    return covs, p_values, lbs, ubs