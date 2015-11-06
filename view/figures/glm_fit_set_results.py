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
plt.style.use('ggplot')

import axis_tools
import insect_glm_fit_helper as igfh

from db_api.connect import session
from db_api import models

EXPERIMENT = 'mosquito_0.4mps_checkerboard_floor'
ODOR_STATE = 'on'
FIT_NAME = 'short_long_timescale_bases_fit_0'

EXAMPLE_TRIAL = 0
EXAMPLE_TRAIN = 0
EXAMPLE_TEST = 0


FIG_SIZE_EXAMPLES = (15, 15)
FIG_SIZE_FILTERS = (10, 15)
FIG_SIZE_RESIDUALS = (10, 15)

FACE_COLOR = 'w'
FONT_SIZE = 16


glm_fit_set = session.query(models.GlmFitSet).filter_by(
    name=FIT_NAME,
    experiment_id=EXPERIMENT,
    odor_state=ODOR_STATE
).first()

start_time_point = glm_fit_set.start_time_point
delay = glm_fit_set.delay

# plot example trajectories
fig, axs = plt.subplots(
    glm_fit_set.n_glms, 2,
    figsize=FIG_SIZE_EXAMPLES,
    facecolor=FACE_COLOR,
    tight_layout=True
)

axs_odor = [[None, None] for _ in range(len(axs))]

glms = glm_fit_set.glms[EXAMPLE_TRIAL]
traj_train_id = glm_fit_set.trajs_train[EXAMPLE_TRIAL][EXAMPLE_TRAIN]
traj_test_id = glm_fit_set.trajs_test[EXAMPLE_TRIAL][EXAMPLE_TEST]

traj_train = session.query(models.Trajectory).get(traj_train_id)
traj_test = session.query(models.Trajectory).get(traj_test_id)

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

axs[0, 0].set_title('Training Trajectory')
axs[0, 1].set_title('Test Trajectory')

for ax in axs.flatten():
    axis_tools.set_fontsize(ax, FONT_SIZE)

for ax in axs_odor.flatten():
    axis_tools.set_fontsize(ax, FONT_SIZE)

plt.show()