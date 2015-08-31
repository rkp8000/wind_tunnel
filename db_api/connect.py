"""
Make a connection to either the analysis database or the analysis testing database.
"""
from __future__ import print_function, division
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


TEST = True
COMMIT = False


def commit(sesh):
    if COMMIT:
        sesh.commit()


if TEST:
    print('CONNECTED TO WIND TUNNEL TEST DATABASE')

    engine = create_engine(os.environ['WIND_TUNNEL_TEST_DB_CXN_URL'])
    figure_data_env_var = 'WIND_TUNNEL_TEST_FIGURE_DATA_DIRECTORY'

else:
    print('CONNECTED TO WIND TUNNEL PRODUCTION DATABASE')

    x = raw_input('Are you sure you want to connect to the production database [y or n]?')
    if x.lower() == 'y':
        engine = create_engine(os.environ['WIND_TUNNEL_DB_CXN_URL'])
    else:
        raise RuntimeError('User prevented write access to database.')

    figure_data_env_var = 'WIND_TUNNEL_FIGURE_DATA_DIRECTORY'

engine.connect()

Session = sessionmaker(bind=engine)
session = Session()
