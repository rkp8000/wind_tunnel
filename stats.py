"""
Some statistical functions.
"""
from __future__ import print_function, division
import numpy as np
from scipy import stats
from sklearn import linear_model


def nansem(x, axis=None):
    """
    Calculate the standard error of the mean ignoring nans.
    :param x: data array
    :param axis: what axis to calculate the sem over
    :return: standard error of the mean
    """

    std = np.nanstd(x, axis=axis, ddof=1)
    sqrt_n = np.sqrt((~np.isnan(x)).sum(axis=axis))

    return std / sqrt_n


def pearsonr_with_confidence(x, y, confidence=0.95):
    """
    Calculate the pearson correlation coefficient, its p-value, and upper and
        lower 95% confidence bound.
    :param x: one array
    :param y: other array
    :param confidence: how confident the confidence interval
    :return: correlation, p-value, lower confidence bound, upper confidence bound
    """

    rho, p = stats.pearsonr(x, y)
    n = len(x)

    # calculate confidence interval on correlation
    # how confident do we want to be?
    n_sds = stats.norm.ppf(1 - (1 - confidence) / 2)
    z = 0.5 * np.log((1 + rho) / (1 - rho))  # convert to z-space
    sd = np.sqrt(1. / (n - 3))
    lb_z = z - n_sds * sd
    ub_z = z + n_sds * sd
    # convert back to rho-space
    lb = (np.exp(2*lb_z) - 1) / (np.exp(2*lb_z) + 1)
    ub = (np.exp(2*ub_z) - 1) / (np.exp(2*ub_z) + 1)

    return rho, p, lb, ub


def pearsonr_difference_significance(r_1, n_1, r_2, n_2):
    """
    Calculate the two-tailed p-value that results from comparing two (Pearson) correlation coefficients.
    Translated into python from code at www.quantpsy.org/corrtest/corrtest.htm.
    :param r_1: first correlation coefficient
    :param n_1: number of data points used to calculate r_1
    :param r_2: second correlation coefficient
    :param n_2: number of data points used to calculate r_2
    :return: one-tailed p-value, two-tailed p-value
    """

    z_1 = 0.5 * (np.log(1 + r_1) - np.log(1 - r_1))
    z_2 = 0.5 * (np.log(1 + r_2) - np.log(1 - r_2))
    z = (z_1 - z_2) / np.sqrt((1 / (n_1 - 3)) + 1 / (n_2 - 3))

    return 2*stats.norm.cdf(-np.abs(z))


def pearsonr_partial_with_confidence(x, y, conditioned_on, confidence=0.95):
    """
    Calculate the correlation of x and y', where y' is the difference between y
    and the best fit hyperplane of y vs. the variables in conditioned_on.
    :param x: input
    :param y: output
    :param conditioned_on: list of variables to condition y on before finding correlation
    :param confidence: confidence interval of partial correlation
    :return: correlation, p-value, lower confidence bound, upper confidence bound
    """
    # calculate y-prime
    inputs = np.array(conditioned_on).T  # format such that each row is a data point, each col a variable
    clf = linear_model.LinearRegression()
    clf.fit(inputs, y)
    y_prime = y - clf.predict(inputs)

    return pearsonr_with_confidence(x, y_prime, confidence=confidence)


def cov_with_confidence(x, y, confidence=0.95):
    """
    Calculate the covariance of two variables, its p-value, and upper and
        lower 95% confidence bound.
    :param x: one array
    :param y: other array
    :param confidence: how confident the confidence interval
    """

    cov = np.cov(x, y)[0, 1]

    corr, pv, lb, ub = pearsonr_with_confidence(x, y, confidence)

    scale_factor = cov / corr

    return cov, pv, lb * scale_factor, ub * scale_factor


def f_test(rss_reduced, rss_full, df_reduced, df_full, n):
    """
    Calculate the F-statistic between a full and nested model.
    :param rss_reduced: residual sum of squares (RSS) for reduced model
    :param rss_full: RSS for full model
    :param df_reduced: degrees of freedom for reduced model
    :param df_full: degrees of freedom for full model
    :param n: number of data points
    :return: F, p-val
    """

    num = (rss_reduced - rss_full) / (df_full - df_reduced)
    denom = rss_full / (n - df_full)

    f = num / denom

    p_val = 1 - stats.f.cdf(f, df_full - df_reduced, n - df_full)

    return f, p_val