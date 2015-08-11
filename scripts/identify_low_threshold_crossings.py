"""
Identify threshold crossings in flight data for low thresholds.
"""
from __future__ import print_function, division
import numpy as np
import time_series
from db_api import models
from db_api.connect import session, commit

THRESHOLD_VALUES = {'fruit_fly': (0.01, 0.1), 'mosquito': (401, 410)}
ODOR_STATES = ('on', 'none', 'afterodor')


def main():

    for th_ctr in range(2):
        for expt in session.query(models.Experiment):
            print('Experiment "{}"'.format(expt.id))

            threshold_value = THRESHOLD_VALUES[expt.insect][th_ctr]

            # make threshold
            threshold = models.Threshold(experiment=expt,
                                         determination='arbitrary',
                                         value=threshold_value)
            session.add(threshold)

            # loop over odor states
            for odor_state in ODOR_STATES:
                print('Odor "{}"'.format(odor_state))

                # make crossing group
                cg_id = '{}_{}_th{}'.format(expt.id, odor_state, threshold_value)
                cg = models.CrossingGroup(id=cg_id,
                                          experiment=expt,
                                          odor_state=odor_state,
                                          threshold=threshold)
                session.add(cg)

                # get crossings for each trajectory
                for traj in session.query(models.Trajectory).\
                    filter_by(experiment=expt, odor_state=odor_state, clean=True):

                    segments, peaks = time_series.segment_by_threshold(traj.odors(session),
                                                                       threshold_value,
                                                                       traj.timepoint_ids_extended)

                    # add crossings
                    for s_ctr, (segment, peak) in enumerate(zip(segments, peaks)):
                        crossing = models.Crossing(trajectory=traj,
                                                   crossing_number=s_ctr + 1,
                                                   crossing_group=cg)
                        crossing.start_timepoint_id = segment[0]
                        crossing.entry_timepoint_id = segment[1]
                        crossing.peak_timepoint_id = segment[2]
                        crossing.exit_timepoint_id = segment[3] - 1
                        crossing.end_timepoint_id = segment[4] - 1
                        crossing.max_odor = peak
                        session.add(crossing)

                    commit(session)


if __name__ == '__main__':
    main()