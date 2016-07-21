"""
Calculate the autocorrelation function for each experiment, odor state, and variable.
"""
from __future__ import print_function, division
from __future__ import print_function, division
import os
import numpy as np
import time_series
from db_api.connect import session, commit, figure_data_env_var
from db_api import models

N_LAGS = 500  # 5 seconds
DIRECTORY_PATH = os.path.join('kinematics', 'timepoint_autocorrelations')
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
QUANTITIES = ['odor']


def main():

    for expt in session.query(models.Experiment):
        print('In experiment "{}"...'.format(expt.id))

        for odor_state in ODOR_STATES:
            print('Odor state = "{}"'.format(odor_state))

            trajs = session.query(models.Trajectory).\
                filter_by(experiment=expt, odor_state=odor_state, clean=True)

            for variable in QUANTITIES:
                print('{}...'.format(variable))

                tp_data = [traj.timepoint_field(session, variable) for traj in trajs]
                n_data_points = np.sum([len(d) for d in tp_data])
                window_len = N_LAGS / expt.sampling_frequency

                acor, p_value, conf_lb, conf_ub = \
                    time_series.xcov_multi_with_confidence(tp_data, tp_data, 0, N_LAGS, normed=True)

                time_vector = np.arange(len(acor)) / expt.sampling_frequency

                file_name = '{}_{}_{}.pickle'.format(expt.id, odor_state, variable)

                tp_acor = models.TimepointAutocorrelation(figure_root_path_env_var=figure_data_env_var,
                                                          directory_path=DIRECTORY_PATH,
                                                          file_name=file_name,
                                                          variable=variable,
                                                          experiment_id=expt.id,
                                                          odor_state=odor_state,
                                                          n_data_points=n_data_points,
                                                          n_trajectories=len(tp_data),
                                                          window_len=window_len)
                tp_acor.data = {'time_vector': time_vector,
                                'autocorrelation': acor,
                                'p_value': p_value,
                                'confidence_lower': conf_lb,
                                'confidence_upper': conf_ub}
                session.add(tp_acor)

                commit(session)


if __name__ == '__main__':
    main()