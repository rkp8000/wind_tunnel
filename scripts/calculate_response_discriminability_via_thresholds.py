"""
Calculate how discriminable behavioral responses time-locked to peaks of plume-crossings are
when they are separated by thresholding the value of the peak odor stimulus during crossing.
"""
from __future__ import print_function, division
import numpy as np
from db_api import models
from db_api.connect import session, commit
import stats

RESPONSE_VAR = 'velocity_a'
DISCRIMINATION_THRESHOLD_VALUES = {'fruit_fly': np.exp(np.linspace(np.log(0.5), np.log(300), 80)),
                                   'mosquito': np.exp(np.linspace(np.log(415), np.log(2000), 80))}
TIME_AVG_START = 0  # relative to peak (in timepoints, 100/s)
TIME_AVG_END = 100
TIME_AVG_REL_TO = 'peak'


def get_time_avg_response_diff_and_bounds(responses_below, responses_above):
    """
    Calculate time-averaged response (e.g., heading) difference between response_below and
    response_above matrices.

    In calculating the difference response_above is subtracted from response_below.
    If either responses_below or responses_above has only one response, the lower and upper bounds
    are returned as Nones.

    :param responses_below: 2D array of responses "below threhsold", can have nans
    :param responses_above: 2D array of responses "above threhsold", can have nans
    :return: difference, lower bound on difference, upper bound on difference
    """

    # get means
    mean_below = np.nanmean(responses_below, axis=0)
    mean_above = np.nanmean(responses_above, axis=0)

    # zero means to first entry
    mean_below -= mean_below[0]
    mean_above -= mean_above[0]

    # get standard errors of the mean
    sem_below = stats.nansem(responses_below, axis=0)
    sem_above = stats.nansem(responses_above, axis=0)

    # get mean difference
    diff = np.nanmean((mean_below - mean_above))

    # get bounds
    if len(responses_below) in [0, 1] or len(responses_above) in [0, 1]:
        lb = None
        ub = None
    else:
        lb = np.nanmean((mean_below - sem_below) - (mean_above + sem_above))
        ub = np.nanmean((mean_below + sem_below) - (mean_above - sem_above))

    return diff, lb, ub


def main():

    n_timesteps = TIME_AVG_END - TIME_AVG_START

    for expt in session.query(models.Experiment):
        for cg in session.query(models.CrossingGroup).\
            filter(models.CrossingGroup.experiment == expt).\
            filter(models.CrossingGroup.odor_state == 'on').\
            filter(models.Threshold.determination == 'arbitrary'):
            print('Crossings group: "{}"'.format(cg.id))

            for th_val in DISCRIMINATION_THRESHOLD_VALUES[expt.insect]:

                crossings_below = session.query(models.Crossing).\
                    filter(models.Crossing.crossing_group == cg).\
                    filter(models.Crossing.max_odor < th_val).all()
                crossings_above = session.query(models.Crossing).\
                    filter(models.Crossing.crossing_group == cg).\
                    filter(models.Crossing.max_odor >= th_val).all()

                responses_below = np.nan * np.ones((len(crossings_below), n_timesteps), dtype=float)
                responses_above = np.nan * np.ones((len(crossings_above), n_timesteps), dtype=float)

                # fill in values
                for crossing, response in zip(crossings_below, responses_below):
                    response_var = crossing.timepoint_field(session, RESPONSE_VAR,
                                                            first=TIME_AVG_START,
                                                            last=TIME_AVG_END - 1,
                                                            first_rel_to=TIME_AVG_REL_TO,
                                                            last_rel_to=TIME_AVG_REL_TO)
                    response[:len(response_var)] = response_var

                for crossing, response in zip(crossings_above, responses_above):
                    response_var = crossing.timepoint_field(session, RESPONSE_VAR,
                                                            first=TIME_AVG_START,
                                                            last=TIME_AVG_END - 1,
                                                            first_rel_to=TIME_AVG_REL_TO,
                                                            last_rel_to=TIME_AVG_REL_TO)
                    response[:len(response_var)] = response_var

                diff, lb, ub = get_time_avg_response_diff_and_bounds(responses_below,
                                                                     responses_above)

                if len(crossings_below) == 0 or len(crossings_above) == 0:
                    diff = None
                    lb = None
                    ub = None

                disc_th = models.DiscriminationThreshold(crossing_group=cg,
                                                         odor_threshold=th_val,
                                                         n_crossings_below=len(crossings_below),
                                                         n_crossings_above=len(crossings_above),
                                                         time_avg_start=TIME_AVG_START,
                                                         time_avg_end=TIME_AVG_END,
                                                         time_avg_rel_to=TIME_AVG_REL_TO,
                                                         variable=RESPONSE_VAR,
                                                         time_avg_difference=diff,
                                                         lower_bound=lb,
                                                         upper_bound=ub)

                session.add(disc_th)
                commit(session)


if __name__ == '__main__':
    main()