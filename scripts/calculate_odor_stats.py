"""
Calculate the integrated odor value for each trajectory
"""
from __future__ import print_function, division

from db_api.connect import session, commit
from db_api import models

MOSQUITO_BASELINE_ODOR = 400


def main():

    for expt in session.query(models.Experiment):
        if 'mosquito' in expt.id:
            baseline = MOSQUITO_BASELINE_ODOR
        else:
            baseline = 0

        trajs = session.query(models.Trajectory).filter_by(experiment=expt, clean=True)
        for traj in trajs:
            odor = traj.odors(session)
            integrated_odor = (odor - baseline).sum() / 100
            traj.odor_stats = models.TrajectoryOdorStats(integrated_odor=integrated_odor)

            session.add(traj)

        commit(session)


if __name__ == '__main__':
    main()