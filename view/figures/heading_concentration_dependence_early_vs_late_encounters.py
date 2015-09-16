"""
Plot the partial correlation of concentration and subsequent heading (conditioned on
heading and upwind/downwind position) at various times for all encounters, as well as
early and late subsets of encounters.
"""
from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np

import stats
import axis_tools
from db_api import models
from db_api.connect import session

SAVE_PATH = '/Users/rkp/Desktop'

DETERMINATION = 'chosen0'  # which group of crossing thresholds to use

N_TIMESTEPS_BEFORE = 0
N_TIMESTEPS_AFTER = 100
TRIGGER = 'peak'

MIN_POSITION_X = 0
MAX_POSITION_X = 0.7
MIN_HEADING_XYZ = 60
MAX_HEADING_XYZ = 120


FACE_COLOR = 'w'
FIG_SIZE = (7, 18)
FONT_SIZE = 20
LW = 2
ALPHA = 0.3

EXPT_COLORS = {'fruitfly_0.3mps_checkerboard_floor': 'b',
               'fruitfly_0.4mps_checkerboard_floor': 'g',
               'fruitfly_0.6mps_checkerboard_floor': 'r'}

EARLY_LATE_COLORS = {'early': 'b',
                     'late': 'g'}

AXES = {'fruitfly_0.3mps_checkerboard_floor': 1,
        'fruitfly_0.4mps_checkerboard_floor': 2,
        'fruitfly_0.6mps_checkerboard_floor': 3}


fig, axs = plt.subplots(
    4, 1, facecolor=FACE_COLOR, figsize=FIG_SIZE, sharex=True, sharey=True,
    tight_layout=True
)

wind_speed_handles = []
wind_speed_labels = []

for expt in session.query(models.Experiment):
    if 'mosquito' in expt.id:
        continue
    print(expt.id)

    # get threshold and crossing group for this experiment
    threshold = session.query(models.Threshold).filter_by(
        experiment=expt, determination=DETERMINATION
    ).first()
    crossing_group = session.query(models.CrossingGroup).filter_by(
        threshold=threshold, odor_state='on'
    ).first()

    # create querysets
    crossings_all = session.query(models.Crossing).filter_by(crossing_group=crossing_group)
    crossings_early = crossings_all.filter(models.Crossing.crossing_number <= 2)
    crossings_late = crossings_all.filter(models.Crossing.crossing_number > 2)

    labels = ['all', 'early', 'late']
    querysets = [crossings_all, crossings_early, crossings_late]

    early_late_handles = []
    early_late_labels = []

    # loop through querysets
    for label, queryset in zip(labels, querysets):
        print(label)
        initial_hs = []
        initial_xs = []
        concs = []
        headings = []

        # get all initial headings, initial xs, peak concentrations, and heading time-series
        for crossing in queryset:
            # throw away crossings that do not meet trigger criteria
            position_x = getattr(crossing.feature_set_basic, 'position_x_{}'.format(TRIGGER))
            heading_xyz = getattr(crossing.feature_set_basic, 'heading_xyz_{}'.format(TRIGGER))

            if not (MIN_POSITION_X <= position_x <= MAX_POSITION_X):
                continue
            if not (MIN_HEADING_XYZ <= heading_xyz <= MAX_HEADING_XYZ):
                continue

            concs.append(crossing.max_odor)
            initial_xs.append(position_x)
            initial_hs.append(heading_xyz)
            headings.append(
                crossing.timepoint_field(
                    session, 'heading_xyz', -N_TIMESTEPS_BEFORE, N_TIMESTEPS_AFTER - 1,
                    TRIGGER, TRIGGER, nan_pad=True
                )
            )

        initial_hs = np.array(initial_hs)
        initial_xs = np.array(initial_xs)
        concs = np.array(concs)
        headings = np.array(headings)

        # calculate partial correlations between concentration and heading at all timepoints,
        # conditioned on h_xyz and x at the time of encounter-trigger

        correlations = np.nan * np.ones((headings.shape[1],), dtype=float)
        p_values = np.nan * np.ones(correlations.shape, dtype=float)
        lbs = np.nan * np.ones(correlations.shape, dtype=float)
        ubs = np.nan * np.ones(correlations.shape, dtype=float)
        ns = np.nan * np.ones(correlations.shape, dtype=float)

        for ctr in range(headings.shape[1]):
            hs = headings[:, ctr]

            # create not-nan mask
            mask = ~np.isnan(hs)
            ns[ctr] = mask.sum()

            # get partial correlations using all not-nan values
            r, p, lb, ub = stats.pearsonr_partial_with_confidence(
                concs[mask],
                hs[mask],
                [initial_hs[mask], initial_xs[mask]],
            )

            correlations[ctr] = r
            p_values[ctr] = p
            lbs[ctr] = lb
            ubs[ctr] = ub

        t = np.arange(-N_TIMESTEPS_BEFORE, N_TIMESTEPS_AFTER) / 100

        if label == 'all':
            # plot on first plot with correct color
            handle, = axs[0].plot(
                t, correlations, color=EXPT_COLORS[expt.id], lw=LW
            )
            axs[0].fill_between(
                t, lbs, ubs, color=EXPT_COLORS[expt.id], alpha=ALPHA
            )
            wind_speed_handles.append(handle)
            wind_speed_labels.append('{} m/s'.format(expt.wind_speed))
        else:
            handle, = axs[AXES[expt.id]].plot(
                t, correlations, color=EARLY_LATE_COLORS[label], lw=LW
            )
            axs[AXES[expt.id]].fill_between(
                t, lbs, ubs, color=EARLY_LATE_COLORS[label], alpha=ALPHA
            )

            early_late_handles.append(handle)
            early_late_labels.append(label)

    axs[AXES[expt.id]].legend(early_late_handles, early_late_labels)

axs[0].legend(wind_speed_handles, wind_speed_labels)

axs[0].set_title('Concentration/heading\npartial correlations')
for ax, label in zip(axs[1:], wind_speed_labels):
    ax.set_title(label)

axs[-1].set_xlabel('Time since encounter {} (s)'.format(TRIGGER))
[ax.set_ylabel('Partial correlation') for ax in axs]

[axis_tools.set_fontsize(ax, FONT_SIZE) for ax in axs]

fig.savefig(
    '{}/conc_heading_partial_correlation_triggered_on_{}_hxyz_between_{}_and_{}.png'.
    format(SAVE_PATH, TRIGGER, MIN_HEADING_XYZ, MAX_HEADING_XYZ)
)