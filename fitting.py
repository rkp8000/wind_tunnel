"""
A module that makes it a bit easier to fit things.
"""
from __future__ import division, print_function
import copy
import numpy as np
import statsmodels.api as sm
import time_series

PLOT_COLOR_CYCLE = ['k', 'b', 'g', 'r', 'c']


class GLMFitter(object):

    def __init__(self, family):
        self.family = family

        self.tsm = time_series.Munger()
        self.constant = None
        self.in_filters = None
        self.out_filter = None
        self.model = None
        self.results = None
        self.feature_matrix = None
        self.response_vector = None

    def set_params(self, delay, basis_in, basis_out):
        self.tsm.delay = delay
        self.original_basis_in = basis_in
        self.original_basis_out = basis_out
        self.tsm.basis_in = copy.deepcopy(basis_in)
        self.tsm.basis_out = copy.deepcopy(basis_out)

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
        self.constant, self.in_filters, self.out_filter = self.tsm.filters_from_coeffs(self.results.params)

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

    def plot_filters(self, ax, x_lim=None, y_lim=None):
        """
        Plot recovered filters on a supplied axis.
        :param ax: axis instance
        :param x_lim: x limits (tuple)
        :param y_lim: y limits (tuple)
        """
        ax.set_title('constant = {}'.format(self.constant))
        for ctr, in_filter in enumerate(self.in_filters):
            if not isinstance(in_filter, float):
                t = np.arange(len(in_filter))
                ax.plot(t, in_filter, color=PLOT_COLOR_CYCLE[ctr + 1], lw=2)

        if isinstance(self.out_filter, float):
            ax.scatter(0, self.out_filter, s=30, c=PLOT_COLOR_CYCLE[-1])
        else:
            t = np.arange(len(self.out_filter))
            ax.plot(t, self.out_filter, color=PLOT_COLOR_CYCLE[-1], lw=2)

        if x_lim:
            ax.set_xlim(x_lim)
        if y_lim:
            ax.set_ylim(y_lim)

    def plot_basis(self, ax, x_lim=None, y_lim=None):
        """
        Plot filter basis functions on a supplied axis.
        :params same as in plot_filters
        """
        for ctr, basis_in in enumerate(self.original_basis_in):
            if basis_in is not None:
                t = np.arange(len(basis_in))
                for basis in basis_in.T:
                    ax.plot(t, basis, color=PLOT_COLOR_CYCLE[ctr + 1], lw=2)

        if self.original_basis_out is not None:
            t = np.arange(len(self.original_basis_out))
            for basis in self.original_basis_out.T:
                ax.plot(t, basis, color=PLOT_COLOR_CYCLE[-1], lw=2)

        if x_lim:
            ax.set_xlim(x_lim)
        if y_lim:
            ax.set_ylim(y_lim)