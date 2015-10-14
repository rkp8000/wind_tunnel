"""
A module that makes it a bit easier to fit things.
"""
from __future__ import division, print_function
import numpy as np
import statsmodels.api as sm
import time_series


class GLMFitter(object):

    def __init__(self, family):
        self.family = family

        self.tsm = time_series.Munger()
        self.constant = None
        self.in_filters = None
        self.out_filters = None
        self.model = None
        self.results = None
        self.feature_matrix = None
        self.response_vector = None

    def set_params(self, delay, basis_in, basis_out):
        self.tsm.delay = delay
        self.tsm.basis_in = basis_in
        self.tsm.basis_out = basis_out

        self.tsm.orthogonalize_basis()

    def make_feature_matrix_and_response_vector(self, data, start):
        """
        Make feature matrices and response vectors from multiple time-series.
        :return: feature matrix and response vector
        """
        feature_matrix = []
        response_vector = []

        for in_data, out_data in data:
            feature_matrix_current, response_vector_current = self.tsm.munge(in_data, out_data, start)
            feature_matrix.append(feature_matrix_current)
            response_vector.append(response_vector_current)

        return np.concatenate(feature_matrix, axis=0), np.concatenate(response_vector, axis=0)

    def fit(self, data, start):
        """
        Fit the model to some data.
        :param data: list of tuples, each corresponding to a trial; each tuple has two elements,
        the first of which is a list of all the input time-series (each of those being a 1D array), and the
        second of which is the output time-series (another 1D array); all inputs and output must have same
        length
        :param start: the index of the first element to start fitting with; e.g., if your filters defined
        over 20 timesteps, you might want to have start = 20
        """
        fm, rv = self.make_feature_matrix_and_response_vector(data, start)

        self.model = sm.GLM(endog=rv, exog=fm, family=self.family)
        self.results = self.model.fit()
        self.constant, self.in_filters, self.out_filters = self.tsm.filters_from_coeffs(self.results.params)

        # save things for later
        self.feature_matrix, self.response_vector = fm, rv

    def predict(self, data=None, start=None):
        """
        Make a prediction.
        :param data: data (list of tuples of input and output time-series); if None, previously created
            feature matrix is used
        :return:
        """
        if data is None:
            feature_matrix = self.feature_matrix
        else:
            feature_matrix, _ = self.make_feature_matrix_and_response_vector(data, start)

        return self.model.predict(self.results.params, exog=feature_matrix)

