"""
Identify all plume-crossings in a specific set of infotaxis simulations.
"""
from __future__ import division, print_function
from scipy.ndimage import gaussian_filter1d as smooth

from db_api.infotaxis import models
from db_api.infotaxis.connect import session
from db_api.infotaxis import add_script_execution

import time_series


HEADING_SMOOTHING = 3
THRESHOLDS = {'fly': 10, 'mosq': 430}

SCRIPTID = 'identify_plume_crossings'


def main(SIM_PREFIX=None, sim_ids=None, thresholds=None, trial_limit=None):
    
    if thresholds is None:
        thresholds = THRESHOLDS
        
    SCRIPTNOTES = ('Identify plume crossings for simulations with prefix "{}" '
        'using heading smoothing "{}" and thresholds "{}"'.format(
        SIM_PREFIX, HEADING_SMOOTHING, thresholds))

    if sim_ids is None:
        SIM_SUFFIXES = [
            'fruitfly_0.3mps_checkerboard_floor_odor_on',
            'fruitfly_0.3mps_checkerboard_floor_odor_none',
            'fruitfly_0.3mps_checkerboard_floor_odor_afterodor',
            'fruitfly_0.4mps_checkerboard_floor_odor_on',
            'fruitfly_0.4mps_checkerboard_floor_odor_none',
            'fruitfly_0.4mps_checkerboard_floor_odor_afterodor',
            'fruitfly_0.6mps_checkerboard_floor_odor_on',
            'fruitfly_0.6mps_checkerboard_floor_odor_none',
            'fruitfly_0.6mps_checkerboard_floor_odor_afterodor',
            'mosquito_0.4mps_checkerboard_floor_odor_on',
            'mosquito_0.4mps_checkerboard_floor_odor_none',
            'mosquito_0.4mps_checkerboard_floor_odor_afterodor',]

        sim_ids = [
            '{}_{}'.format(SIM_PREFIX, sim_suffix)
            for sim_suffix in SIM_SUFFIXES
        ]

    # add script execution to database
    add_script_execution(
        SCRIPTID, session=session, multi_use=True, notes=SCRIPTNOTES)

    for sim_id in sim_ids:

        print('Identifying crossings from simulation: "{}"'.format(sim_id))

        # get simulation

        sim = session.query(models.Simulation).filter_by(id=sim_id).first()

        # get all trials from this simulation

        trials = session.query(models.Trial).filter_by(simulation=sim).all()

        # make crossing group

        if 'fly' in sim_id:

            threshold = thresholds['fly']

        elif 'mosq' in sim_id:

            threshold = thresholds['mosq']

        cg_id = '{}_th_{}_hsmoothing_{}'.format(
            sim_id, threshold, HEADING_SMOOTHING)
        
        print('Storing in crossing group:')
        print(cg_id)

        cg = models.CrossingGroup(
            id=cg_id,
            simulation=sim,
            threshold=threshold,
            heading_smoothing=HEADING_SMOOTHING)

        session.add(cg)

        # loop through trials and identify crossings

        trial_ctr = 0

        for trial in trials:

            if trial_limit and trial_ctr >= trial_limit:

                break

            # get relevant time-series

            odors = trial.timepoint_field(session, 'odor')

            xs = trial.timepoint_field(session, 'xidx')
            ys = trial.timepoint_field(session, 'yidx')
            zs = trial.timepoint_field(session, 'zidx')

            # get smoothed headings

            hs = smooth(trial.timepoint_field(session, 'hxyz'), HEADING_SMOOTHING)

            # identify crossings

            crossing_lists, peaks = time_series.segment_by_threshold(
                odors, threshold)

            tr_start = trial.start_timepoint_id

            # add crossings

            for c_ctr, (crossing_list, peak) in enumerate(zip(crossing_lists, peaks)):

                crossing = models.Crossing(
                    trial=trial,
                    crossing_number=c_ctr+1,
                    crossing_group=cg,
                    start_timepoint_id=crossing_list[0] + tr_start,
                    entry_timepoint_id=crossing_list[1] + tr_start,
                    peak_timepoint_id=crossing_list[2] + tr_start,
                    exit_timepoint_id=crossing_list[3] + tr_start - 1,
                    end_timepoint_id=crossing_list[4] + tr_start - 1,
                    max_odor=peak,)

                session.add(crossing)

                # create this crossing's basic feature set

                crossing.feature_set_basic = models.CrossingFeatureSetBasic(
                    position_x_entry=xs[crossing_list[1]],
                    position_y_entry=ys[crossing_list[1]],
                    position_z_entry=zs[crossing_list[1]],
                    heading_xyz_entry=hs[crossing_list[1]],
                    position_x_peak=xs[crossing_list[2]],
                    position_y_peak=ys[crossing_list[2]],
                    position_z_peak=zs[crossing_list[2]],
                    heading_xyz_peak=hs[crossing_list[2]],
                    position_x_exit=xs[crossing_list[3] - 1],
                    position_y_exit=ys[crossing_list[3] - 1],
                    position_z_exit=zs[crossing_list[3] - 1],
                    heading_xyz_exit=hs[crossing_list[3] - 1],
                )

                session.add(crossing)

            trial_ctr += 1

        # commit after all crossings from all trials from a simulation have been added

        session.commit()


if __name__ == '__main__':

    main()
