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

SAVE_PATH = '/Users/rkp/Desktop/wind_tunnel_figs'

DETERMINATION = 'chosen0'  # which group of crossing thresholds to use

N_TIMESTEPS_BEFORE = 20
N_TIMESTEPS_AFTER = 100
TRIGGER = 'peak'

MIN_POSITION_X = 0
MAX_POSITION_X = 0.7
MIN_HEADING_XYZ = 0  # 60
MAX_HEADING_XYZ = 180  # 120

FACE_COLOR = 'w'
FIG_SIZE = (16, 18)
FONT_SIZE = 20
LEGEND_FONT_SIZE = 14
LW = 2
ALPHA = 0.3

EXPT_COLORS = {'fruitfly_0.3mps_checkerboard_floor': 'b',
               'fruitfly_0.4mps_checkerboard_floor': 'g',
               'fruitfly_0.6mps_checkerboard_floor': 'r',
               'mosquito_0.4mps_checkerboard_floor': 'k',}

EARLY_LATE_COLORS = {'early': 'b',
                     'late': 'g'}

AXES = {'fruitfly_0.3mps_checkerboard_floor': 2,
        'fruitfly_0.4mps_checkerboard_floor': 3,
        'fruitfly_0.6mps_checkerboard_floor': 4,
        'mosquito_0.4mps_checkerboard_floor': 5,}


fig, axs_unflat = plt.subplots(
    3, 2, facecolor=FACE_COLOR, figsize=FIG_SIZE, sharex=True,
    tight_layout=True
)

axs = axs_unflat.flatten()
axs_twin = [ax.twinx() for ax in axs[2:]]

wind_speed_handles = []
wind_speed_labels = []

for expt in session.query(models.Experiment):

    if 'fly' in expt.id:

        wind_speed_label = 'fly {} m/s'.format(expt.wind_speed)

    elif 'mosquito' in expt.id:

        wind_speed_label = 'mosquito {} m/s'.format(expt.wind_speed)

    print(expt.id)

    # get threshold and crossing group for this experiment
    threshold = session.query(models.Threshold).filter_by(
        experiment=expt, determination=DETERMINATION
    ).first()

    crossing_group = session.query(models.CrossingGroup).filter_by(
        threshold=threshold, odor_state='none'
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
            wind_speed_labels.append(wind_speed_label)

            # plot p-value
            axs[1].plot(
                t, p_values, color=EXPT_COLORS[expt.id], lw=LW
            )

        else:
            handle, = axs[AXES[expt.id]].plot(
                t, correlations, color=EARLY_LATE_COLORS[label], lw=LW
            )
            axs[AXES[expt.id]].fill_between(
                t, lbs, ubs, color=EARLY_LATE_COLORS[label], alpha=ALPHA
            )

            early_late_handles.append(handle)
            early_late_labels.append(label)

            if label == 'early':
                # store these for later so we can compare them to the lates
                early_correlations = correlations
                early_ns = ns
            elif label == 'late':
                # calculate significance between early and late correlations
                p_vals = []
                for r_1, n_1, r_2, n_2 in zip(early_correlations, early_ns, correlations, ns):
                    p_vals.append(stats.pearsonr_difference_significance(r_1, n_1, r_2, n_2))

                ax = axs_twin[AXES[expt.id] - 2]
                ax.plot(t, p_vals, c='k', ls='-', lw=2)
                ax.axhline(0.05, c='k', ls='--')
                ax.set_ylim(0, 0.5)

    axs_twin[AXES[expt.id] - 2].legend(early_late_handles, early_late_labels, loc='best')

axs[0].legend(wind_speed_handles, wind_speed_labels, loc='best')

axs[0].set_title('Concentration/heading\npartial correlations')

axs[1].set_ylim(0, 1)
axs[1].legend(wind_speed_handles, wind_speed_labels, loc='best')
axs[1].set_title('P-values')


for ax, label in zip(axs[2:], wind_speed_labels):

    ax.set_title(label)

for ax in axs:

    ax.set_xlabel('Time since encounter {} (s)'.format(TRIGGER))

[ax.set_ylabel('Partial correlation') for ax in axs]
axs[1].set_ylabel('p value')
[ax.set_ylabel('P-value') for ax in axs_twin]

[axis_tools.set_fontsize(ax, FONT_SIZE, LEGEND_FONT_SIZE) for ax in axs]
[axis_tools.set_fontsize(ax, FONT_SIZE, LEGEND_FONT_SIZE) for ax in axs_twin]

fig.savefig(
    '{}/conc_heading_partial_correlation_triggered_on_{}_hxyz_between_{}_and_{}_{}_control.png'.
    format(SAVE_PATH, TRIGGER, MIN_HEADING_XYZ, MAX_HEADING_XYZ, DETERMINATION)
)