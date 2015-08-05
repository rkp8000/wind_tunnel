"""
Functions for playing with time-serieses.
"""
from __future__ import print_function, division
import numpy as np
from numpy import concatenate as cc


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