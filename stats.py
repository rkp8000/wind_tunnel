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


def pearsonr_partial_with_confidence(x, y, conditioned_on, confidence=0.95):
    """
    Calculate the partial correlation of x and y', where y' is the difference between y
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