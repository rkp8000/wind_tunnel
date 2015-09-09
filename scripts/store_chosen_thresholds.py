"""
Store a set of thresholds in the database, each with determination "chosen{}".format(N),
where N indicates which set of thresholds to use (should these change).
"""
from __future__ import print_function, division
from db_api.connect import session, commit
from db_api import models

THRESHOLDS_DICT = {
    'fruitfly_0.3mps_checkerboard_floor': 10,
    'fruitfly_0.4mps_checkerboard_floor': 10,
    'fruitfly_0.6mps_checkerboard_floor': 10,
    'mosquito_0.4mps_checkerboard_floor': 430,
}


def main():

    for expt_id, value in THRESHOLDS_DICT.iteritems():
        threshold = models.Threshold(experiment_id=expt_id,
                                     determination='chosen0',
                                     value=value)
        session.add(threshold)
        commit(session)


if __name__ == '__main__':
    main()