"""
Code to make figure showing heading time-series triggered on plume-exit.
"""
from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp as ks_test
import stats
import axis_tools
from db_api import models
from db_api.connect import session

SAVE_PATH = '/Users/rkp/Desktop'

DETERMINATION = 'chosen0'  # which group of crossing thresholds to use

N_TIMESTEPS_BEFORE = 20
N_TIMESTEPS_AFTER = 80
TRIGGER = 'peak'

MIN_POSITION_X = 0
MAX_POSITION_X = 0.7
MIN_HEADING_XYZ = 60
MAX_HEADING_XYZ = 120

BINS_POS_X = np.linspace(MIN_POSITION_X, MAX_POSITION_X, 30)
BINS_HEADING = np.linspace(0, 180, 30)
EXAMPLE_HEADING_TIME_START = 0.55  # s
EXAMPLE_HEADING_TIME_END = 0.65  # s

FIG_SIZE = (14, 18)
FIG_SIZE_POS = (10, 14)
FACE_COLOR = 'white'
FONT_SIZE = 20

EXPT_COLORS = {'fruitfly_0.3mps_checkerboard_floor': 'b',
               'fruitfly_0.4mps_checkerboard_floor': 'g',
               'fruitfly_0.6mps_checkerboard_floor': 'r'}

EARLY_LATE_COLORS = {'early': 'b',
                     'late': 'g'}

AXES = {'fruitfly_0.3mps_checkerboard_floor': 1,
        'fruitfly_0.4mps_checkerboard_floor': 2,
        'fruitfly_0.6mps_checkerboard_floor': 3}


fig, axs = plt.subplots(
    4, 2, figsize=FIG_SIZE, facecolor=FACE_COLOR, sharex=True, sharey=True, tight_layout=True
)
fig_tp, axs_tp = plt.subplots(
    3, 1, figsize=FIG_SIZE_POS, facecolor=FACE_COLOR, sharex=True, sharey=True, tight_layout=True
)
fig_pos, axs_pos = plt.subplots(
    3, 1, figsize=FIG_SIZE_POS, facecolor=FACE_COLOR, sharex=True, tight_layout=True
)

wind_speed_handles = []
wind_speed_labels = []

for expt_ctr, expt in enumerate(session.query(models.Experiment)):
    if 'mosquito' in expt.id:
        continue

    print(expt.id)
    wind_speed_labels.append('{} m/s'.format(expt.wind_speed))
    subset_handles = []
    subset_labels = []
    subset_handles_pos = []
    subset_labels_pos = []
    subset_handles_heading = []
    subset_labels_heading = []

    threshold = session.query(models.Threshold).filter_by(
        experiment=expt, determination=DETERMINATION
    ).first()

    crossing_group_on = session.query(models.CrossingGroup).\
        filter_by(threshold=threshold, odor_state='on').first()
    crossing_group_none = session.query(models.CrossingGroup).\
        filter_by(threshold=threshold, odor_state='none').first()

    crossings_on = session.query(models.Crossing).filter_by(crossing_group=crossing_group_on)

    crossings_none = session.query(models.Crossing).filter_by(crossing_group=crossing_group_none)

    for col_ctr, crossings in enumerate([crossings_on, crossings_none]):
        # make time vector
        time_vec = np.arange(-N_TIMESTEPS_BEFORE, N_TIMESTEPS_AFTER) / 100
        example_tp_start = np.where(time_vec == EXAMPLE_HEADING_TIME_START)[0][0]
        example_tp_end = np.where(time_vec == EXAMPLE_HEADING_TIME_END)[0][0]
        # collect all time series
        crossings_time_series = np.nan * np.ones(
            (len(crossings.all()), N_TIMESTEPS_BEFORE + N_TIMESTEPS_AFTER)
        )

        crossing_ctr = 0
        for crossing in crossings:

            position_x = getattr(crossing.feature_set_basic, 'position_x_{}'.format(TRIGGER))
            heading_xyz = getattr(crossing.feature_set_basic, 'heading_xyz_{}'.format(TRIGGER))

            if not (MIN_POSITION_X <= position_x <= MAX_POSITION_X):
                continue
            if not (MIN_HEADING_XYZ <= heading_xyz <= MAX_HEADING_XYZ):
                continue

            headings = crossing.timepoint_field(
                session, 'heading_xyz', -N_TIMESTEPS_BEFORE, N_TIMESTEPS_AFTER - 1,
                TRIGGER, TRIGGER, nan_pad=True
            )

            crossings_time_series[crossing_ctr, :] = headings
            crossing_ctr += 1

        crossings_time_series = crossings_time_series[:crossing_ctr, :]

        print('{} total crossings'.format(len(crossings_time_series)))

        # get mean and sem
        crossings_mean = np.nanmean(crossings_time_series, axis=0)
        crossings_sem = stats.nansem(crossings_time_series, axis=0)

        # make plot
        handle, = axs[0, col_ctr].plot(time_vec, crossings_mean, lw=2, color=EXPT_COLORS[expt.id])
        axs[0, col_ctr].fill_between(
            time_vec, crossings_mean - crossings_sem, crossings_mean + crossings_sem,
            color=EXPT_COLORS[expt.id], alpha=0.3
        )

        if col_ctr == 0:
            wind_speed_handles.append(handle)

        # make querysets for early and late crossings
        crossings_early = crossings.filter(models.Crossing.crossing_number <= 2)
        crossings_late = crossings.filter(models.Crossing.crossing_number > 2)

        for label, crossings_subset in zip(['early', 'late'], [crossings_early, crossings_late]):

            # allocate list/space for crossing positions and time-series
            crossing_positions = []
            crossings_time_series = np.nan * np.ones(
                (len(crossings_subset.all()), N_TIMESTEPS_BEFORE + N_TIMESTEPS_AFTER)
            )

            # loop through all crossings
            crossing_ctr = 0
            for crossing in crossings_subset:

                # throw away crossings that do not meet criteria
                position_x = getattr(crossing.feature_set_basic, 'position_x_{}'.format(TRIGGER))
                heading_xyz = getattr(crossing.feature_set_basic, 'heading_xyz_{}'.format(TRIGGER))

                if not (MIN_POSITION_X <= position_x <= MAX_POSITION_X):
                    continue
                if not (MIN_HEADING_XYZ <= heading_xyz <= MAX_HEADING_XYZ):
                    continue

                # store heading time-series
                headings = crossing.timepoint_field(
                    session, 'heading_xyz', -N_TIMESTEPS_BEFORE, N_TIMESTEPS_AFTER - 1,
                    TRIGGER, TRIGGER, nan_pad=True
                )
                crossings_time_series[crossing_ctr, :] = headings

                # store x position
                crossing_positions.append(position_x)

                crossing_ctr += 1

            # truncate the crossing_time_series array to the number of selected crossings
            crossings_time_series = crossings_time_series[:crossing_ctr, :]

            print('{} crossings in {} subset'.format(len(crossings_time_series), label))

            # get mean and sem of crossing time-series
            crossings_mean = np.nanmean(crossings_time_series, axis=0)
            crossings_sem = stats.nansem(crossings_time_series, axis=0)

            # make plots
            color = EARLY_LATE_COLORS[label]
            handle, = axs[AXES[expt.id], col_ctr].plot(
                time_vec, crossings_mean, lw=2, color=color
            )
            axs[AXES[expt.id], col_ctr].fill_between(
                time_vec, crossings_mean - crossings_sem, crossings_mean + crossings_sem,
                lw=2, color=color, alpha=0.3
            )
            axs[AXES[expt.id], col_ctr].axvspan(
                EXAMPLE_HEADING_TIME_START, EXAMPLE_HEADING_TIME_END, color='gray', alpha=0.3
            )

            if col_ctr == 0:
                subset_handles.append(handle)
                subset_labels.append(label)

            # plot example heading distributions and position distributions at time of trigger
            if col_ctr == 0:
                example_headings = np.nanmean(
                    crossings_time_series[:, example_tp_start:example_tp_end], axis=1
                )
                if label == 'early':
                    example_headings_early = example_headings
                    positions_early = crossing_positions[:]
                elif label == 'late':
                    example_headings_late = example_headings
                    _, p_val_heading = ks_test(example_headings_early, example_headings_late)
                    positions_late = crossing_positions[:]
                    _, p_val_pos = ks_test(positions_early, positions_late)

                # plot histograms
                # heading
                cts, bins = np.histogram(example_headings, bins=BINS_HEADING, normed=True)
                bincs = 0.5 * (bins[:-1] + bins[1:])
                handle, = axs_tp[AXES[expt.id] - 1].plot(bincs, cts, lw=2, color=color)
                subset_handles_heading.append(handle)
                mean_str = '{0:.2f}'.format(np.round(np.nanmean(example_headings), 3))
                label_heading = '{}, mean = {} deg'.format(label, mean_str)
                if label == 'late':
                    label_heading += ', p = {0:.3f} (KS test)'.format(p_val_heading)
                subset_labels_heading.append(label_heading)

                # position
                cts, bins = np.histogram(crossing_positions, bins=BINS_POS_X, normed=True)
                bincs = 0.5 * (bins[:-1] + bins[1:])
                handle, = axs_pos[AXES[expt.id] - 1].plot(bincs, cts, lw=2, color=color)
                subset_handles_pos.append(handle)
                mean_str = '{0:.2f}'.format(np.round(np.mean(crossing_positions), 3))
                label_pos = '{}, mean = {}m'.format(label, mean_str)
                if label == 'late':
                    label_pos += ', p = {0:2.3} (KS test)'.format(p_val_pos)
                subset_labels_pos.append(label_pos)

    # make legends
    axs[AXES[expt.id], 0].legend(subset_handles, subset_labels)
    axs_tp[AXES[expt.id] - 1].legend(subset_handles_heading, subset_labels_heading)
    axs_pos[AXES[expt.id] - 1].legend(subset_handles_pos, subset_labels_pos)

# clean heading time-series plot
axs[0, 0].set_title('Ethanol')
axs[0, 1].set_title('Control')

# title wind-speed-specific early/late plots
for ax_row, label in zip(axs[1:], wind_speed_labels):
    [ax.set_title(label) for ax in ax_row]

# make wind-speed-overlay plot's legend
axs[0, 0].legend(wind_speed_handles, wind_speed_labels)

# set x and y labels and font size
[ax.set_xlabel('Time since {} (s)'.format(TRIGGER)) for ax in axs[-1, :]]
[ax.set_ylabel('Heading (deg)') for ax in axs[:, 0]]
[axis_tools.set_fontsize(ax, FONT_SIZE) for ax in axs.flatten()]

# save heading time-series plot
fig.savefig(
    '{}/crossings_triggered_on_{}_hxyz_between_{}_and_{}.png'.
    format(SAVE_PATH, TRIGGER, MIN_HEADING_XYZ, MAX_HEADING_XYZ)
)

# clean up heading histogram plots
# title wind-speed-specific early/late plots
[ax.set_title(label) for ax, label in zip(axs_tp, wind_speed_labels)]

# set x and y labels and font size
axs_tp[-1].set_xlabel('uw/dw position at {} (m)'.format(TRIGGER))
[ax.set_ylabel('probability') for ax in axs_tp]
[axis_tools.set_fontsize(ax, FONT_SIZE) for ax in axs_tp.flatten()]

# save position histograms plot
fig_tp.savefig(
    '{}/crossings_triggered_on_{}_hxyz_between_{}_and_{}_heading_hist.png'.
    format(SAVE_PATH, TRIGGER, MIN_HEADING_XYZ, MAX_HEADING_XYZ)
)

# clean up position histogram plots
# title wind-speed-specific early/late plots
[ax.set_title(label) for ax, label in zip(axs_pos, wind_speed_labels)]

# set x and y labels and font size
axs_pos[-1].set_xlabel('uw/dw position at {} (m)'.format(TRIGGER))
[ax.set_ylabel('probability') for ax in axs_pos]
[axis_tools.set_fontsize(ax, FONT_SIZE) for ax in axs_pos.flatten()]

# save position histograms plot
fig_pos.savefig(
    '{}/crossings_triggered_on_{}_hxyz_between_{}_and_{}_position_hist.png'.
    format(SAVE_PATH, TRIGGER, MIN_HEADING_XYZ, MAX_HEADING_XYZ)
)