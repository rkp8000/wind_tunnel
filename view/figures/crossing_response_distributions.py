"""
Show the distributions of crossing-triggered behavioral time-series, time-locked to peak odor
during plume crossing.
"""
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import stats
from db_api import models
from db_api.connect import session

THRESHOLD_ID = 2  # look up in threshold table
DISCRIMINATION_THRESHOLD = 20
DISPLAY_START = -50
DISPLAY_END = 150
INTEGRAL_START = 0  # timepoints (fs 100 Hz) relative to peak
INTEGRAL_END = 100
VARIABLE = 'heading_xyz'

FACE_COLOR = 'white'
FIG_SIZE = (10, 10)
LW = 2
COLORS = ['k', 'r']  # [below, above] discrimination threshold


# get threshold and crossing group for odor on trajectories
threshold = session.query(models.Threshold).get(THRESHOLD_ID)
cg = session.query(models.CrossingGroup).\
    filter_by(threshold=threshold, odor_state='on').first()

all_crossings = session.query(models.Crossing).\
    filter(models.Crossing.crossing_group == cg).all()

# get all crossings where max odor is below/above discrimination threshold
crossings_below = session.query(models.Crossing).\
    filter(models.Crossing.crossing_group == cg).\
    filter(models.Crossing.max_odor < DISCRIMINATION_THRESHOLD).all()
crossings_above = session.query(models.Crossing).\
    filter(models.Crossing.crossing_group == cg).\
    filter(models.Crossing.max_odor >= DISCRIMINATION_THRESHOLD).all()

# make array for storing crossing-specific time-series
n_timepoints = DISPLAY_END - DISPLAY_START
time_vec = np.arange(DISPLAY_START, DISPLAY_END)
var_array_below = np.nan * np.ones((len(crossings_below), n_timepoints), dtype=float)
var_array_above = np.nan * np.ones((len(crossings_above), n_timepoints), dtype=float)

# fill in arrays
for crossings, var_array in zip([crossings_below, crossings_above],
                                [var_array_below, var_array_above]):
    for ctr, crossing in enumerate(crossings):

        var_before = crossing.timepoint_field(session, VARIABLE, first=DISPLAY_START, last=-1,
                                              first_rel_to='peak', last_rel_to='peak')
        var_after = crossing.timepoint_field(session, VARIABLE, first=0, last=DISPLAY_END - 1,
                                             first_rel_to='peak', last_rel_to='peak')

        var_array[ctr, -DISPLAY_START - len(var_before):-DISPLAY_START] = var_before
        var_array[ctr, -DISPLAY_START:-DISPLAY_START + len(var_after)] = var_after

# calculate mean, std, and sem of these arrays
mean_below = np.nanmean(var_array_below, axis=0)
std_below = np.nanstd(var_array_below, axis=0)
sem_below = stats.nansem(var_array_below, axis=0)

mean_above = np.nanmean(var_array_above, axis=0)
std_above = np.nanstd(var_array_above, axis=0)
sem_above = stats.nansem(var_array_above, axis=0)

# get integrals
integral_mean = mean_above[INTEGRAL_START - DISPLAY_START:INTEGRAL_END - DISPLAY_START] - \
    mean_below[INTEGRAL_START - DISPLAY_START:INTEGRAL_END - DISPLAY_START]

# plot things nicely
fig, ax = plt.subplots(1, 1, facecolor=FACE_COLOR, figsize=FIG_SIZE, tight_layout=True)

ax.plot(time_vec, mean_below, color=COLORS[0], lw=LW)
ax.plot(time_vec, mean_below - std_below, color=COLORS[0], lw=LW, ls='--')
ax.plot(time_vec, mean_below + std_below, color=COLORS[0], lw=LW, ls='--')
ax.fill_between(time_vec, mean_below - sem_below, mean_below + sem_below,
                color=COLORS[0], lw=LW, alpha=0.3)

ax.plot(time_vec, mean_above, color=COLORS[1], lw=LW)
ax.plot(time_vec, mean_above - std_above, color=COLORS[1], lw=LW, ls='--')
ax.plot(time_vec, mean_above + std_above, color=COLORS[1], lw=LW, ls='--')
ax.fill_between(time_vec, mean_above - sem_above, mean_above + sem_above,
                color=COLORS[1], lw=LW, alpha=0.3)

ax.axvline(INTEGRAL_START, ls='--')
ax.axvline(INTEGRAL_END, ls='--')

ax.set_xlabel('time step')
ax.set_ylabel(VARIABLE)

ax.set_title('{}_discrimination_{}'.format(threshold.experiment.id, DISCRIMINATION_THRESHOLD))

print('Red = above threshold')
print('Black = below threhsold')
print('Integral = {}'.format(integral_mean.sum()))

plt.show()