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


def make_exponential_basis_functions(input_taus, output_taus, domain_factor=4.61):
    """
    Make a set of exponential basis functions given the relevant timescales.
    :param input_taus: list of input timescales, each element being a tuple of timescales (or None if no basis)
    :param output_taus: list of output timescales, each element being a tuple of timescales
    :param domain_factor: how long filter domain should be relative to timescale of filter
    :return:
    """

    basis_ins = []  # i.e., one for each model
    basis_outs = []

    for input_tau, output_tau in zip(input_taus, output_taus):
            basis_in = []
            for tau_set in input_tau:
                if tau_set is None:
                    basis = None
                else:
                    t = np.arange(np.round(domain_factor * np.max(tau_set)), dtype=float)
                    basis = np.zeros((len(t), len(tau_set)), dtype=float)
                    for ctr, tau in enumerate(tau_set):
                        basis[:, ctr] = np.exp(-t / tau)
                basis_in.append(basis)

            if output_tau is None:
                basis_out = None
            else:
                t = np.arange(np.round(domain_factor * np.max(output_tau)), dtype=float)
                basis_out = np.zeros((len(t), len(output_tau)), dtype=float)
                for ctr, tau in enumerate(output_tau):
                    basis_out[:, ctr] = np.exp(-t / tau)

            basis_ins.append(basis_in)
            basis_outs.append(basis_out)

    # figure out max filter length
    max_filter_length = 0
    for basis_in, basis_out in zip(basis_ins, basis_outs):
        in_filter_max = np.max([0] + [len(basis) for basis in basis_in if basis is not None])
        if basis_out is not None:
            out_filter_max = len(basis_out)
        else:
            out_filter_max = 0
        filter_max = max(in_filter_max, out_filter_max)
        if filter_max > max_filter_length:
            max_filter_length = filter_max

    return basis_ins, basis_outs, max_filter_length