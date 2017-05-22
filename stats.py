"""
Some statistical functions.
"""
from __future__ import print_function, division
import numpy as np
from scipy import stats
from scipy.stats import distributions
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


def partial_corr(x, y, controls):
    """
    Calculate the partial correlation of x and y conditioned on a set of
    control variables. This is calculated by first determining
    :param x: 1D array
    :param y: 1D array
    :param controls: list of 1D arrays
    :return: partial correlation coefficient, p-value
    """

    for control in controls: assert len(control) == len(x) == len(y)

    control_array = np.array(controls).T
    residuals = []

    for v in [x, y]:

        lrg = linear_model.LinearRegression()
        lrg.fit(control_array, v)
        residuals.append(v - lrg.predict(control_array))

    return stats.pearsonr(residuals[0], residuals[1])


def _chk2_asarray(a, b, axis):
    if axis is None:
        a = np.ravel(a)
        b = np.ravel(b)
        outaxis = 0
    else:
        a = np.asarray(a)
        b = np.asarray(b)
        outaxis = axis
    return a, b, outaxis


def _ttest_finish(df,t):
    """Common code between all 3 t-test functions."""
    prob = distributions.t.sf(np.abs(t), df) * 2  # use np.abs to get upper tail
    if t.ndim == 0:
        t = t[()]

    return t, prob


def ttest_adjusted_ns(a, b, n1, n2, axis=0, equal_var=True):
    """
    Calculate a t-test except with different n's from the number of samples.
    """
    a, b, axis = _chk2_asarray(a, b, axis)
    if a.size == 0 or b.size == 0:
        return (np.nan, np.nan)

    v1 = np.var(a, axis, ddof=1)
    v2 = np.var(b, axis, ddof=1)

    if (equal_var):
        df = n1 + n2 - 2
        svar = ((n1 - 1) * v1 + (n2 - 1) * v2) / float(df)
        denom = np.sqrt(svar * (1.0 / n1 + 1.0 / n2))
    else:
        vn1 = v1 / n1
        vn2 = v2 / n2
        df = ((vn1 + vn2) ** 2) / ((vn1 ** 2) / (n1 - 1) + (vn2 ** 2) / (n2 - 1))

        # If df is undefined, variances are zero (assumes n1 > 0 & n2 > 0).
        # Hence it doesn't matter what df is as long as it's not NaN.
        df = np.where(np.isnan(df), 1, df)
        denom = np.sqrt(vn1 + vn2)

    d = np.mean(a, axis) - np.mean(b, axis)
    t = np.divide(d, denom)
    t, prob = _ttest_finish(df, t)

    return t, prob
