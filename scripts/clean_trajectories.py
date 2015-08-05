"""
Clean insect flight trajectories.
"""
from __future__ import print_function, division
from datetime import datetime
import numpy as np
from db_api.connect import session, commit
from db_api import models
import time_series

INSECTS = ['fruit_fly', 'mosquito']
DATETIME_FORMAT = '%Y%m%d_%H%M%S'


def clean_traj(traj, cleaning_params):
    """
    Return the start and end timepoints of the clean portions of a trajectory.

    :param traj: a Trajectory
    :param cleaning_params: cleaning parameter dictionary
    :return: list of tuples giving start and end timepoint ids
    """
    min_speed_threshold = cleaning_params['min_speed_threshold']
    dist_to_wall_threshold = cleaning_params['dist_to_wall_threshold']
    min_pause_threshold = cleaning_params['min_pause_threshold']
    min_trajectory_length = cleaning_params['min_trajectory_length']

    # get relevant trajectory information
    stp_id, etp_id = traj.start_timepoint_id, traj.end_timepoint_id
    speeds = traj.velocities_a(session)
    dists_from_wall = traj.distances_from_wall(session)

    # get mask of all timepoints below speed and distance from wall threshold
    paused_mask = (speeds < min_speed_threshold) * (dists_from_wall < dist_to_wall_threshold)

    # get start and end idxs of pauses
    pause_starts, pause_ends = time_series.segment_basic(paused_mask)

    # set paused_mask elements to false when pause is below minimum duration
    for pd_ctr, pause_duration in enumerate(pause_ends - pause_starts):
        if pause_duration < min_pause_threshold:
            paused_mask[pause_starts[pd_ctr]:pause_ends[pd_ctr]] = False

    # get starts and ends of active portions of trajectories
    clean_portions = np.transpose(time_series.segment_basic(~paused_mask, t=np.arange(stp_id, etp_id + 1)))

    # decrement end timepoint ids since SQL uses inclusive ends
    clean_portions[:, 1] -= 1

    # return only portions that are sufficiently long
    return clean_portions[(clean_portions[:, 1] - clean_portions[:, 0]) > min_trajectory_length]


def make_basic_trajectory_info(traj):
    """
    Create a model for this trajectory's basic info.
    :param traj: a Trajectory
    :return: TrajectoryBasicInfo instance
    """

    positions = traj.positions(session)
    start_x, start_y, start_z = positions[0]
    end_x, end_y, end_z = positions[-1]

    dists = traj.distances_from_wall(session)
    dist_start = dists[0]
    dist_end = dists[-1]

    duration = len(positions) / traj.experiment.sampling_frequency

    start_datetime = datetime.strptime(traj.id[1:16], DATETIME_FORMAT)

    tbi = models.TrajectoryBasicInfo(trajectory=traj,
                                     start_position_x=start_x,
                                     start_position_y=start_y,
                                     start_position_z=start_z,
                                     end_position_x=end_x,
                                     end_position_y=end_y,
                                     end_position_z=end_z,
                                     dist_from_wall_start=dist_start,
                                     dist_from_wall_end=dist_end,
                                     start_datetime=start_datetime,
                                     duration=duration)

    return tbi


def main():

    for insect in INSECTS:
        cleaning_params_queryset = session.query(models.TrajectoryCleaningParameter).filter_by(insect=insect)
        cleaning_params = dict(cleaning_params_queryset.all())

        for expt in session.query(models.Experiment).filter_by(insect=insect):
            for traj in expt.trajs:

                clean_portions = clean_traj(traj, cleaning_params)

                for ctr, clean_portion in enumerate(clean_portions):

                    if clean_portion[0] == traj.start_timepoint_id and clean_portion[1] == traj.end_timepoint_id:
                        traj.clean = True
                        portion_traj = traj

                    else:
                        stp_id, etp_id = clean_portion
                        # make new trajectory
                        id = traj.id + '_c{}'.format(ctr)
                        portion_traj = model.Trajectory(id=id,
                                                        start_timepoint_id=stp_id,
                                                        end_timepoint_id=etp_id,
                                                        experiment=expt,
                                                        raw=False,
                                                        clean=True,
                                                        odor_state=expt.odor_state)
                    session.add(portion_traj)
                    portion_traj.basic_info = make_trajectory_basic_info(portion_traj)
                    session.add(portion_traj)

                commit(session)