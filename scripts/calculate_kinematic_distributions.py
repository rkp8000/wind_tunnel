"""
Calculate distributions of kinematic quantities.
"""
from __future__ import print_function, division
import os
import numpy as np
from db_api.connect import session, commit, figure_data_env_var
from db_api import models

N_BINS = 100
DIRECTORY_PATH = os.path.join('kinematics', 'timepoint_distributions')
ODOR_STATES = ['on', 'none', 'afterodor']
QUANTITIES = ['position_x',
              'position_y',
              'position_z',
              'velocity_x',
              'velocity_y',
              'velocity_z',
              'velocity_a',
              'acceleration_x',
              'acceleration_y',
              'acceleration_z',
              'acceleration_a',
              'heading_xy',
              'heading_xz',
              'heading_xyz',
              'angular_velocity_x',
              'angular_velocity_y',
              'angular_velocity_z',
              'angular_velocity_a',
              'angular_acceleration_x',
              'angular_acceleration_y',
              'angular_acceleration_z',
              'angular_acceleration_a',
              'distance_from_wall']


def make_distribution(data, n_bins, lb=None, ub=None):
    """
    Calculate the distribution of a given variable.
    :param data: list of 1D arrays of possibly different sizes containing variable of interest
    :param n_bins: number of bins to use
    :param lb: lower bound (lower edge of lowest bin)
    :param ub: upper bound (upper edge of highest bin)
    :return: counts, bins
    """
    m = np.nanmean(data)
    s = np.nanstd(data)

    if lb is None:
        lb = m - 5 * s

    if ub is None:
        ub = m + 5 * s

    bins = np.linspace(lb, ub, n_bins + 1)

    return np.histogram(data, bins)


def main():

    for expt in session.query(models.Experiment):
        print('In experiment "{}"...'.format(expt.id))

        for odor_state in ODOR_STATES:
            print('Odor state = "{}"'.format(odor_state))

            trajs = session.query(models.Trajectory).\
                filter_by(experiment=expt, odor_state=odor_state, clean=True)

            for variable_name in QUANTITIES:
                print('{}...'.format(variable_name))

                traj_data = []

                traj_ctr = 0
                for traj in trajs:

                    traj_data.extend(traj.timepoint_field(session, variable_name))

                    traj_ctr += 1

                lb, ub = None, None

                if variable_name.endswith('_a') or 'heading' in variable_name:
                    lb = 0
                    if 'heading' in variable_name:
                        ub = 180

                cts, bins = make_distribution(np.array(traj_data), N_BINS, lb=lb, ub=ub)

                file_name = '{}_{}_{}.pickle'.format(expt.id, odor_state, variable_name)

                tp_dstr = models.TimepointDistribution(figure_root_path_env_var=figure_data_env_var,
                                                       directory_path=DIRECTORY_PATH,
                                                       file_name=file_name,
                                                       variable=variable_name,
                                                       experiment_id=expt.id,
                                                       odor_state=odor_state,
                                                       n_data_points=len(traj_data),
                                                       n_trajectories=traj_ctr,
                                                       bin_min=bins[0],
                                                       bin_max=bins[-1],
                                                       n_bins=N_BINS)
                tp_dstr.data = {'cts': cts, 'bins': bins}
                session.add(tp_dstr)

                commit(session)


if __name__ == '__main__':
    main()