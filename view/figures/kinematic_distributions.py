"""
View set of kinematic distributions.
"""
from __future__ import print_function, division
from itertools import product as cproduct
import matplotlib.pyplot as plt
from db_api.connect import session
from db_api import models
from plot import set_font_size


def show_distributions(EXPERIMENTS, VARIABLES, ODOR_STATES, AX_GRID):

    AX_SIZE = (6, 4)
    LW = 2
    COLORS = ('b', 'g', 'r')

    fig_size = (AX_SIZE[0] * AX_GRID[1], AX_SIZE[1] * AX_GRID[0])

    fig, axs = plt.subplots(*AX_GRID,
                            figsize=fig_size,
                            tight_layout=True)

    for ax, (expt_id, variable) in zip(axs.flatten(), cproduct(EXPERIMENTS, VARIABLES)):

        handles = []

        for odor_state, color in zip(ODOR_STATES, COLORS):

            tp_dstr = session.query(models.TimepointDistribution).filter_by(
                variable=variable, experiment_id=expt_id,
                odor_state=odor_state).first()

            handles.append(ax.plot(
                tp_dstr.bincs, tp_dstr.cts, lw=LW, color=color, label=odor_state)[0])

        ax.set_xlabel(variable)
        ax.set_ylabel('counts')

        ax.legend(handles=handles)

        ax.set_title('{}\n{}'.format(expt_id, variable))

    for ax in axs.flatten():

        set_font_size(ax, 16)

    return fig
