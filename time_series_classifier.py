"""
Code for classifying vector time-series.
"""
from __future__ import print_function, division
import numpy as np
from scipy import stats

import kinematics


class VarModel(object):

    def __init__(self, dim, order):
        self.dim = dim
        self.order = order

        self._a = None
        self._k = None
        self.a_full = None
        self.k_inv = None
        self.k_det = None

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, a):
        self._a = a
        # create full a matrix so that entire update can be done in one multiplication
        self.a_full = np.concatenate(a, axis=1)

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, k):
        self._k = k

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

    def log_prob(self, ts):
        """
        Calculate the log probability of observing a set of time-series. The calculation
        begins starting at time points at self.order. (E.g., if it's a 3rd order model, we can
        only calculate the probabilities of time points at index 3 and above).
        :param ts: time-series
        :return: log probability
        """
        self.k_inv = np.linalg.inv(self.k)
        self.k_det = np.linalg.det(self.k)

        # calculate predictions at each time point
        predictors = self.munge(ts, order=self.order)
        predictions = self.a_full.dot(predictors.T)
        truths = ts[self.order:, :].T

        log_probs = self.log_prob_mvn(truths, means=predictions, cov_inv=self.k_inv, cov_det=self.k_det)
        return log_probs.sum()

    def fit(self, tss):
        """
        Fit model to time-series data using analytical max-likelihood solution.
        """
        # build up design matrix (dm) and prediction matrix (d)
        d = np.transpose(np.concatenate([ts[self.order:] for ts in tss], axis=0))
        dm = np.transpose(np.concatenate([self.munge(ts, self.order) for ts in tss], axis=0))

        # recover a_full
        self.a_full = d.dot(dm.T).dot(np.linalg.inv(dm.dot(dm.T)))

        # recover covariance matrix
        self.k = np.cov(d - self.a_full.dot(dm))

        # create list version of a from a_full
        self.a = list(np.split(self.a_full, self.order, axis=1))

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

    @staticmethod
    def munge(data, order):
        """
        Reshape a vector time-series into a matrix each of whose rows is the concatenation of the last
        several vector time points.
        :param data:
        :param order:
        :return:
        """
        n_rows = len(data) - order
        temp = [data[order - offset - 1:order - offset + n_rows - 1, :] for offset in range(0, order)]
        return np.concatenate(temp, axis=1)


class VarClassifierBinary(object):

    def __init__(self, dim, order):
        self.dim = dim
        self.order = order
        self.model_pos = None
        self.model_neg = None
        self.log_prior_pos = None
        self.log_prior_neg = None

    def train(self, positives, negatives):
        """
        Train classifier to be able to distinguish the two sets of time-series.
        :param positives: list of example time-series belonging to positive class
        :param negatives: list of example time-series belonging to negative class
        """
        # learn priors
        prior_pos = len(positives) / (len(positives) + len(negatives))
        self.log_prior_pos = np.log(prior_pos)
        self.log_prior_neg = np.log(1 - prior_pos)

        # fit models
        self.model_pos = VarModel(self.dim, self.order)
        self.model_neg = VarModel(self.dim, self.order)

        self.model_pos.fit(positives)
        self.model_neg.fit(negatives)

    def predict(self, tss):
        """
        Predict which class each time-series belongs to.
        :param tss:
        :return:
        """
        predictions = []

        for ts in tss:

            log_like_pos = self.model_pos.log_prob(ts)
            log_like_neg = self.model_neg.log_prob(ts)

            if (log_like_pos + self.log_prior_pos) > (log_like_neg + self.log_prior_neg):
                predictions.append(1)
            else:
                predictions.append(-1)

        return predictions


class MeanSpeedClassifierBinary(object):
    """
    Distributions over mean speed are represented with Gaussians, for what that's worth.
    """

    def __init__(self):
        self.mean_pos = None
        self.mean_neg = None
        self.std_pos = None
        self.std_neg = None

    def train(self, positives, negatives):
        """
        Train this ubersimple classifier.
        :param positives:
        :param negatives:
        :return:
        """
        speeds_pos = [kinematics.norm(positive).mean() for positive in positives]
        speeds_neg = [kinematics.norm(negative).mean() for negative in negatives]

        self.mean_pos = np.mean(speeds_pos)
        self.mean_neg = np.mean(speeds_neg)

        self.std_pos = np.std(speeds_pos)
        self.std_neg = np.std(speeds_neg)

    def predict(self, tss):
        """
        Make predictions on a set of time-series.
        :param tss:
        :return:
        """
        predictions = []

        for ts in tss:

            mean_speed = kinematics.norm(ts).mean()

            l_pos = stats.norm.pdf(mean_speed, self.mean_pos, self.std_pos)
            l_neg = stats.norm.pdf(mean_speed, self.mean_neg, self.std_neg)

            if l_pos > l_neg:
                predictions.append(1)
            else:
                predictions.append(-1)

        return predictions


class MeanHeadingClassifierBinary(object):
    """
    Distributions over mean heading are represented with Gaussians, for what that's worth.
    """

    def __init__(self):
        self.mean_pos = None
        self.mean_neg = None
        self.std_pos = None
        self.std_neg = None

    def train(self, positives, negatives):
        """
        Train this ubersimple classifier.
        :param positives:
        :param negatives:
        :return:
        """
        headings_pos = [kinematics.heading(positive).mean() for positive in positives]
        headings_neg = [kinematics.heading(negative).mean() for negative in negatives]

        self.mean_pos = np.mean(headings_pos)
        self.mean_neg = np.mean(headings_neg)

        self.std_pos = np.std(headings_pos)
        self.std_neg = np.std(headings_neg)

    def predict(self, tss):
        """
        Make predictions on a set of time-series.
        :param tss:
        :return:
        """
        predictions = []

        for ts in tss:

            mean_heading = kinematics.heading(ts).mean()

            l_pos = stats.norm.pdf(mean_heading, self.mean_pos, self.std_pos)
            l_neg = stats.norm.pdf(mean_heading, self.mean_neg, self.std_neg)

            if l_pos > l_neg:
                predictions.append(1)
            else:
                predictions.append(-1)

        return predictions


class StdHeadingClassifierBinary(object):
    """
    Distributions over mean heading are represented with Gaussians, for what that's worth.
    """

    def __init__(self):
        self.mean_pos = None
        self.mean_neg = None
        self.std_pos = None
        self.std_neg = None

    def train(self, positives, negatives):
        """
        Train this ubersimple classifier.
        :param positives:
        :param negatives:
        :return:
        """
        headings_pos = [kinematics.heading(positive).std() for positive in positives]
        headings_neg = [kinematics.heading(negative).std() for negative in negatives]

        self.mean_pos = np.mean(headings_pos)
        self.mean_neg = np.mean(headings_neg)

        self.std_pos = np.std(headings_pos)
        self.std_neg = np.std(headings_neg)

    def predict(self, tss):
        """
        Make predictions on a set of time-series.
        :param tss:
        :return:
        """
        predictions = []

        for ts in tss:

            mean_heading = kinematics.heading(ts).std()

            l_pos = stats.norm.pdf(mean_heading, self.mean_pos, self.std_pos)
            l_neg = stats.norm.pdf(mean_heading, self.mean_neg, self.std_neg)

            if l_pos > l_neg:
                predictions.append(1)
            else:
                predictions.append(-1)

        return predictions