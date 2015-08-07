"""
Calculate distributions of kinematic quantities.
"""
from __future__ import print_function, division
import os
import numpy as np
from db_api.connect import session, commit
from db_api import models

N_BINS = 100
FIGURE_ROOT_ENV_VARIABLE = 'WIND_TUNNEL_FIGURE_DATA_DIRECTORY'
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


def make_distribution(data, n_bins, bounded_by_zero):
    """
    Calculate the distribution of a given variable.
    :param data: list of 1D arrays of possibly different sizes containing variable of interest
    :param n_bins: number of bins to use
    :param bounded_by_zero: whether or not this quantity has a lower bound of zero
    :return: counts, bins
    """
    m = np.nanmean(data)
    s = np.nanstd(data)

    lb_temp = m - 5 * s

    if bounded_by_zero:
        lb = max(lb_temp, 0)
    else:
        lb = lb_temp

    ub = m + 5 * s

    bins = np.linspace(lb, ub, n_bins + 1)

    return np.histogram(data, bins)


def main():
    figure_root = os.environ[FIGURE_ROOT_ENV_VARIABLE]

    for expt in session.query(models.Experiment):
        print('In experiment ""...'.format(expt.id))

        for odor_state in odor_states:
            print('Odor state = ""'.format(odor_state))

            trajs = session.query(models.Trajectory).\
                filter(models.Trajectory.experiment == expt,
                       models.Trajectory.odor_state == odor_state)

            for variable_name in QUANTITIES:
                traj_data = []

                for traj in trajs:
                    stp_id, etp_id = traj.start_timepoint_id, traj.end_timepoint_id

                    traj_data += list(session.query(models.Timepoint).\
                        filter(getattr(variable_name).between(stp_id, etp_id)).all())

                if variable_name.endswith('_a') or 'heading' in variable_name:
                    bounded_by_zero = True
                else:
                    bounded_by_zero = False

                cts, bins = make_distribution(np.array(traj_data), N_BINS, bounded_by_zero)

                file_name = '{}_{}_{}.pickle'.format(expt.id, odor_state, variable_name)

                tp_dstr = models.TimepointDistribution(figure_root=figure_root,
                                                       directory_path=DIRECTORY_PATH,
                                                       file_name=file_name,
                                                       variable=variable_name,
                                                       experiment_id=expt.id,
                                                       odor_state=odor_state,
                                                       n_data_points=len(traj_data),
                                                       n_trajectories=len(trajs),
                                                       bin_min=bins[0],
                                                       bin_max=bins[-1],
                                                       n_bins=N_BINS)
                tp_dstr.data = {'cts': cts, 'bins': bins}
                session.add(tp_dstr.data)

                commit(session)


if __name__ == '__main__':
    main()