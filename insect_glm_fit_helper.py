from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np

from db_api.connect import session
from db_api import models


def get_trajs_with_integrated_odor_above_threshold(experiment_id, odor_state, integrated_odor_threshold):
    """
    Return all trajectories from a given experiment/odor state that have a certain minimum odor.
    :param experiment_id: experiment id
    :param odor_state: odor state
    :param integrated_odor_threshold: threshold
    :return: list of trajectories
    """
    trajs_all = session.query(models.Trajectory).filter(
        models.Trajectory.experiment_id == experiment_id,
        models.Trajectory.odor_state == odor_state,
        models.Trajectory.clean,
    )
    trajs = []
    for traj in trajs_all:
        if traj.odor_stats.integrated_odor > integrated_odor_threshold:
            trajs.append(traj)

    return trajs


def time_series_from_trajs(trajs, inputs, output):
    """
    Get time-series data from trajectories into a nice list that can be passed to our GLMFitter class.
    :param trajs: list of trajectories
    :param inputs: names of input fields
    :param output: name of output fields
    :return:
    """
    time_series = []
    for traj in trajs:
        input_ts = []
        for input_name in inputs:
            input_ts.append(traj.timepoint_field(session, input_name))
        output_ts = traj.timepoint_field(session, output)

        time_series.append((input_ts, output_ts))

    return time_series