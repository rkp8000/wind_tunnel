from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from db_api.connect import session
from db_api import models

from axis_tools import set_fontsize


def response_discrimination_via_thresholds(
        RESPONSE_VAR,
        CROSSING_GROUP_IDS,
        CROSSING_GROUP_LABELS,
        CROSSING_GROUP_X_LIMS,
        AX_SIZE, FONT_SIZE):
    """
    Plot the difference between the mean time-averaged headings for plume-crossings
    above and below an odor concentration threshold as a function of that threshold.
    """
    fig_size = (AX_SIZE[0], len(CROSSING_GROUP_IDS) * AX_SIZE[1])

    fig, axs = plt.subplots(
        len(CROSSING_GROUP_IDS), 1, facecolor='white',
        figsize=fig_size, tight_layout=True)

    if len(CROSSING_GROUP_IDS) == 1:

        axs = [axs]

    for cg_id, ax in zip(CROSSING_GROUP_IDS, axs):

        cg = session.query(models.CrossingGroup).filter_by(id=cg_id).first()

        disc_ths = session.query(models.DiscriminationThreshold).\
            filter_by(crossing_group=cg, variable=RESPONSE_VAR)

        ths = np.array([disc_th.odor_threshold for disc_th in disc_ths])
        means = np.array([disc_th.time_avg_difference for disc_th in disc_ths], dtype=float)
        lbs = np.array([disc_th.lower_bound for disc_th in disc_ths], dtype=float)
        ubs = np.array([disc_th.upper_bound for disc_th in disc_ths], dtype=float)

        ax.plot(ths, means, color='k', lw=3)
        ax.fill_between(ths, lbs, ubs, color='k', alpha=.3)

        ax.set_xlim(CROSSING_GROUP_X_LIMS[cg.id])

        ax.set_xlabel('threshold')
        ax.set_ylabel('heading change')
        ax.set_title(CROSSING_GROUP_LABELS[cg.id])

    for ax in axs:

        set_fontsize(ax, FONT_SIZE)

    return fig