"""
Store data-cleaning parameters in the database.
"""
from __future__ import print_function, division
from db_api.connect import session, commit
from db_api import models

FRUIT_FLY_PARAMS = {
                    'speed_threshold': 0.03,  # m/s
                    'dist_to_wall_threshold': 0.01,  # m
                    'min_pause_threshold': 10,  # hundredths of a second
                    'min_trajectory_length': 50,  # hundredths of a second
                    }

MOSQUITO_PARAMS = {
                   'speed_threshold': 0.03,  # m/s
                   'dist_to_wall_threshold': 0.01,  # m
                   'min_pause_threshold': 10,  # hundredths of a second
                   'min_trajectory_length': 50,  # hundredths of a second
                   }

for param, value in FRUIT_FLY_PARAMS.items():
    tcp = models.TrajectoryCleaningParameter(insect='fruit_fly',
                                             param=param,
                                             value=value)
    session.add(tcp)

for param, value in MOSQUITO_PARAMS.items():
    tcp = models.TrajectoryCleaningParameter(insect='mosquito',
                                             param=param,
                                             value=value)
    session.add(tcp)

commit(session)