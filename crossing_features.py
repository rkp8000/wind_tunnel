"""
Functions for calculating specific features of crossings.
"""
from __future__ import print_function, division
import numpy as np


def max_odor(crossing, odors, traj_start):
    """
    Calculate the maximum odor during a crossing.
    :param crossing:
    :param odors:
    :param traj_start:
    :return:
    """
    start = crossing.entry_timepoint_id - traj_start
    end = crossing.exit_timepoint_id - traj_start + 1
    return odors[start:end].max()


def mean_odor(crossing, odors, traj_start):
    """
    Calculate the mean odor during a crossing.
    :param crossing:
    :param odors:
    :param traj_start:
    :return:
    """
    start = crossing.entry_timepoint_id - traj_start
    end = crossing.exit_timepoint_id - traj_start + 1
    return odors[start:end].mean()