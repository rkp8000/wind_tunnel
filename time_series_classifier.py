"""
Code for classifying vector time-series.
"""
from __future__ import print_function, division
import numpy as np


class VarModel(object):

    def __init__(self, a, k):
        self.a = a
        self.k = k
        self.noise_mean = np.zeros((len(k),), dtype=float)
        self.order = len(a)
        self.dim = len(k)

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
            next_tp = np.zeros((self.dim, 1))
            for d_ctr in range(self.order):
                next_tp += self.a[d_ctr].dot(ts[ctr - d_ctr, :][:, None])
            ts[ctr] = next_tp.flatten()

        return ts

    def log_prob(self, tss):
        """
        Calculate the log probability of observing a set of time-series. The calculation
        begins starting at time points at self.order. (E.g., if it's a 3rd order model, we can
        only calculate the probabilities of time points at index 3 and above).
        :param tss: list of time-series
        :return: log probability
        """
        pass


class VarClassifierBinary(object):

    def __init__(self):
        pass

    def train(self, positives, negatives):
        pass

    def predict(self, time_series):
        pass

