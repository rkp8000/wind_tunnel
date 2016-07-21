"""
Plot the discriminability of two response sets when they are classified by thresholding.
"""
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from db_api.connect import session
from db_api import models

RESPONSE_VAR = 'heading_xyz'

FACE_COLOR = 'white'
AX_SIZE = (6, 3)


expts = session.query(models.Experiment).all()

fig_size = (2 * AX_SIZE[0], len(expts) * AX_SIZE[1])
fig, axs = plt.subplots(len(expts), 2, facecolor=FACE_COLOR, figsize=fig_size, tight_layout=True)

for expt, ax_row in zip(expts, axs):

    cgs = session.query(models.CrossingGroup).filter_by(experiment=expt, odor_state='on')

    for cg, ax in zip(cgs, ax_row):
        disc_ths = session.query(models.DiscriminationThreshold).\
            filter_by(crossing_group=cg, variable=RESPONSE_VAR)

        ths = np.array([disc_th.odor_threshold for disc_th in disc_ths])
        means = np.array([disc_th.time_avg_difference for disc_th in disc_ths], dtype=float)
        lbs = np.array([disc_th.lower_bound for disc_th in disc_ths], dtype=float)
        ubs = np.array([disc_th.upper_bound for disc_th in disc_ths], dtype=float)

        ax.errorbar(ths, means, [means - lbs, ubs - means], lw=2, elinewidth=1)
        ax.set_title(cg.id)

[ax.set_xlabel('threshold') for ax in axs[-1, :]]
[ax.set_ylabel('heading change') for ax in axs[:, 0]]

plt.show()