"""
View set of kinematic distributions.
"""
from __future__ import print_function, division
import matplotlib.pyplot as plt
from db_api.connect import session
from db_api import models
from config.kinematic_distributions import FIG_LAYOUTS
import plot


FACE_COLOR = 'white'
FONT_SIZE = 16
AX_SIZE = (6, 4)
LW = 2
COLORS = ('b', 'g', 'r')

for fig_layout in fig_layouts:

    fig_size = (AX_SIZE[0], AX_SIZE[1] * len(fig_layout))

    fig, axs = plt.subplots(len(fig_layout), 1,
                            facecolor=FACE_COLOR,
                            figsize=fig_size,
                            tight_layout=True)

    for ax, ax_layout in zip(axs, fig_layout):

        for data_ctr, data_layout in enumerate(ax_layout[0]):

            variable, experiment_id, odor_state = data_layout

            tp_dstr = session.query(models.TimepointDistribution).\
                filter_by(variable=variable, experiment_id=experiment_id, odor_state=odor_state)

            ax.plot(tp_dstr.bincs, tp_dstr.cts, lw=LW, color=COLORS[data_ctr])

        ax.set_xlabel(ax_layout[1][0])
        ax.set_ylabel(ax_layout[1][1])

        plot.set_font_size(ax, FONT_SIZE)

plt.show()