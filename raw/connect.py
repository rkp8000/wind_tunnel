"""
Make connections to the various wind tunnel related databases.

"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


engine = create_engine(os.environ['WIND_TUNNEL_RAW_DB_CXN_URL'])
engine_sample = create_engine(os.environ['WIND_TUNNEL_RAW_SAMPLE_DB_CXN_URL'])

session = sessionmaker(bind=engine)()
session_sample = sessionmaker(bind=engine_sample)()