"""
Code for viewing results of fitting glms to data.

This code makes the following plots:

1. Example training and test trajectory fit by each of a set of glms for one trial.

2. Filters found by each of a set of glms for one trail.

3. Distribution of changes in fit residuals for each successive pair of glms.
"""
from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats
plt.style.use('ggplot')

import axis_tools
import insect_glm_fit_helper as igfh

from db_api.connect import session
from db_api import models

EXPERIMENT = 'fruitfly_0.4mps_checkerboard_floor'
ODOR_STATE = 'on'
FIT_NAME = 'short_long_timescale_bases_fit_1_log'

EXAMPLE_TRIAL = 0
EXAMPLE_TRAIN = 0
EXAMPLE_TEST = 0

N_BINS_RESIDUALS = 20

FIG_SIZE_EXAMPLES = (18, 15)
FIG_SIZE_FILTERS = (14, 15)
FIG_SIZE_RESIDUALS = (14, 15)

FACE_COLOR = 'w'
FONT_SIZE = 16

MODEL_LABELS = (
    'constant',
    'x',
    'x, odor',
    'x, odor_short',
    'x, odor_long',
)


glm_fit_set = session.query(models.GlmFitSet).filter_by(
    name=FIT_NAME,
    experiment_id=EXPERIMENT,
    odor_state=ODOR_STATE
).first()

start_time_point = glm_fit_set.start_time_point
delay = glm_fit_set.delay
n_glms = glm_fit_set.n_glms

# get GLMs for example trial
glms = glm_fit_set.glms[EXAMPLE_TRIAL]

# plot example trajectories
traj_train_id = glm_fit_set.trajs_train[EXAMPLE_TRIAL][EXAMPLE_TRAIN]
traj_test_id = glm_fit_set.trajs_test[EXAMPLE_TRIAL][EXAMPLE_TEST]

traj_train = session.query(models.Trajectory).get(traj_train_id)
traj_test = session.query(models.Trajectory).get(traj_test_id)

fig, axs = plt.subplots(
    n_glms, 2,
    figsize=FIG_SIZE_EXAMPLES,
    facecolor=FACE_COLOR,
    tight_layout=True
)

axs_odor = [[None, None] for _ in range(len(axs))]

for t_ctr, traj in enumerate([traj_train, traj_test]):
    for g_ctr, glm in enumerate(glms):

        ax = axs[g_ctr, t_ctr]
        ax_odor = ax.twinx()

        data = igfh.time_series_from_trajs(
            [traj],
            inputs=glm.input_set,
            output=glm.output
        )

        full_len = len(data[0][1])

        prediction = glm.predict(data=data, start=start_time_point)
        _, ground_truth = glm.make_feature_matrix_and_response_vector(
            data, start_time_point
        )

        odor = igfh.time_series_from_trajs(
            [traj],
            inputs=('odor',),
            output=glm.output
        )[0][0][0][start_time_point + delay:]

        t = np.arange(full_len)[-len(prediction):]
        t_odor = np.arange(start_time_point + delay, full_len)

        ax.plot(t, ground_truth, 'k', lw=2)
        ax.plot(t, prediction, 'r', lw=2)
        ax_odor.plot(t_odor, odor, 'b', lw=2)

        ax.set_xlim(0, full_len)

        axs_odor[g_ctr][t_ctr] = ax_odor

axs_odor = np.array(axs_odor)

for ax in axs[:, 0]:
    ax.set_ylabel(glm_fit_set.predicted)

for ax in axs_odor[:, -1]:
    ax.set_ylabel('odor', color='b')

for ax in axs[-1, :]:
    ax.set_xlabel('time steps')

axs[0, 0].set_title('Training (blue - odor, black - heading, red - predicted heading)')
axs[0, 1].set_title('Test Trajectory')

for ax in axs.flatten():
    axis_tools.set_fontsize(ax, FONT_SIZE)

for ax in axs_odor.flatten():
    axis_tools.set_fontsize(ax, FONT_SIZE)


# plot filters
fig, axs = plt.subplots(
    n_glms, 2,
    facecolor=FACE_COLOR,
    figsize=FIG_SIZE_FILTERS,
    tight_layout=True,
)

for g_ctr, glm in enumerate(glms):
    glm.plot_filters(axs[g_ctr, 0])
    glm.plot_basis(axs[g_ctr, 1])

axs[0, 0].set_title('Filters')
axs[0, 1].set_title('Basis functions')

for ax in axs[-1, :]:
    ax.set_xlabel('time steps')

for ax in axs.flatten():
    axis_tools.set_fontsize(ax, FONT_SIZE)


# plot residuals
residuals = np.array(glm_fit_set.residuals)

fig = plt.figure(
    figsize=FIG_SIZE_RESIDUALS,
    facecolor=FACE_COLOR,
    tight_layout=True,
)

ax_main = fig.add_subplot(1, 2, 1)
axs = [fig.add_subplot(n_glms - 1, 2, (ctr + 1)*2) for ctr in range(n_glms - 1)]

ax_main.plot(residuals.T, color='k', ls='--')
ax_main.set_xticks(range(5))
ax_main.set_xticklabels(MODEL_LABELS, rotation='vertical')

for ctr, resid in enumerate(residuals.T):
    ax_main.scatter(ctr * np.ones(len(resid)), resid, s=50, c='k')

    if ctr >= 1:
        diff = resid - residuals.T[ctr - 1]
        _, p_val = sp_stats.ttest_1samp(diff, 0)

        axs[ctr - 1].hist(diff, N_BINS_RESIDUALS)
        axs[ctr - 1].set_title('M = {0:.4f}, P = {1:.4f}'.format(diff.mean(), p_val))

        axs[ctr - 1].axvline(diff.mean(), ls='--', color='b', lw=2)

ax_main.set_ylabel('Test set prediction error ({})'.format(glm_fit_set.predicted))
axs[-1].set_xlabel('Test prediction difference')

for ax in axs:
    ax.set_ylabel('Counts')

for ax in axs + [ax_main]:
    axis_tools.set_fontsize(ax, FONT_SIZE)

plt.show()