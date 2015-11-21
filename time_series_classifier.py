"""
Code for classifying vector time-series.
"""
from __future__ import print_function, division
import numpy as np


class VarModel(object):

    def __init__(self, a, k):
        self.a = a
        self.k = k
        self.k_inv = np.linalg.inv(k)
        self.k_det = np.linalg.det(k)
        self.noise_mean = np.zeros((len(k),), dtype=float)
        self.order = len(a)
        self.dim = len(k)

        # create full a matrix so that entire update can be done in one multiplication
        self.a_full = np.concatenate(a, axis=1)

    def sample(self, t, initial='zero'):
        """
        Sample a time-series of length t from the model.
        :param t: how many time steps to include
        :param initial: how to set the initial values
        :return: 2D numpy array, rows are time-points
        """

        ts = np.nan * np.zeros((t, self.dim), dtype=float)

        if initial == 'zero':
            ts[:self.order, :] = 0
        else:
            ts[:self.order, :] = initial

        for ctr in range(self.order, t):
            # get concatenated version of previous timesteps
            prev_stack = ts[ctr - self.order:ctr, :]
            prev_cc = np.concatenate(np.flipud(prev_stack))[:, None]
            mean = self.a_full.dot(prev_cc).flatten()
            next_tp = np.random.multivariate_normal(mean, self.k)
            ts[ctr] = next_tp
        return ts

    def log_prob(self, tss):
        """
        Calculate the log probability of observing a set of time-series. The calculation
        begins starting at time points at self.order. (E.g., if it's a 3rd order model, we can
        only calculate the probabilities of time points at index 3 and above).
        :param tss: list of time-series
        :return: log probability
        """
        # calculate predictions at each time point
        predictors = np.concatenate([self.munge(ts, order=self.order) for ts in tss], axis=0)
        predictions = self.a_full.dot(predictors.T)
        truths = np.concatenate([ts[self.order:, :] for ts in tss], axis=0).T

        log_probs = self.log_prob_mvn(truths, means=predictions, cov_inv=self.k_inv, cov_det=self.k_det)
        return log_probs.sum()

    @staticmethod
    def log_prob_mvn(data, means, cov=None, cov_inv=None, cov_det=None):
        """
        Calculate the log-probabilities of a set of multivariate data points given a set of means and a
        covariance matrix.
        :param data: data array - rows are variables, columns are samples
        :param means: array of means, same size as data
        :param cov: covariance matrix
        :param cov_inv: inverse covariance matrix (optional)
        :param cov_det: covariance matrix determinant (optional)
        :return: array of log probabilities, one for each data point
        """
        if cov_inv is None:
            cov_inv = np.linalg.inv(cov)
        if cov_det is None:
            cov_det = np.linalg.det(cov)

        # calculate difference between data and means
        diffs = data - means
        dim = diffs.shape[0]
        # calculate quadratic term of probability
        quad_terms = -(diffs * cov_inv.dot(diffs)).sum(axis=0) / 2
        # calculate normalization factor
        norm_factor = np.log((2*np.pi) ** (-0.5*dim)) + np.log(cov_det ** -0.5)

        return quad_terms + norm_factor


class VarClassifierBinary(object):

    def __init__(self):
        pass

    def train(self, positives, negatives):
        pass

    def predict(self, time_series):
        pass

