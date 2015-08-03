"""
Make a connection to either the analysis database or the analysis testing database.
"""
from __future__ import print_function, division
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


TEST = False