"""
Plot histogram of take-off positions.
"""
from __future__ import print_function, division

FIG_SIZE = (16, 7)
FONT_SIZE = 20
PROJECTION = 'xy'

import numpy as np
import matplotlib.pyplot as plt

from db_api import models
from db_api.connect import session

from config import *
from config.take_off_positions_heatmap import *

row_labels = ('0.3 m/s', '0.4 m/s', '0.6 m/s')
col_labels = ('on', 'none', 'afterodor')

fig, axs = plt.subplots(3, 3, facecolor='white', figsize=FIG_SIZE, tight_layout=True)

for e_ctr, expt in enumerate(EXPERIMENTS):
    for o_ctr, odor_state in enumerate(ODOR_STATES):

        sim_id = SIMULATION_ID.format(expt, odor_state)
        sim = session.query(models.Simulation).get(sim_id)

        sim.analysis_take_off_position_histogram.fetch_data(session)

        if PROJECTION == 'xy':
            heatmap = sim.analysis_take_off_position_histogram.xy
            extent = sim.env.extentxy
            xlabel = 'x'
            ylabel = 'y'
        elif PROJECTION == 'xz':
            heatmap = sim.analysis_take_off_position_histogram.xz
            extent = sim.env.extentxz
            xlabel = 'x'
            ylabel = 'z'
        elif PROJECTION == 'yz':
            heatmap = sim.analysis_take_off_position_histogram.yz
            extent = sim.env.extentyz
            xlabel = 'y'
            ylabel = 'z'

        # calculate entropy
        ax = axs[e_ctr, o_ctr]
        ax.matshow(np.log(heatmap).T, origin='lower', extent=extent)

        # labels
        if e_ctr == 2:
            ax.set_xlabel(xlabel)

        if o_ctr == 0:
            ax.set_ylabel(ylabel)

        ax.set_title('{} {}'.format(row_labels[e_ctr], col_labels[o_ctr]))

plt.show(block=True)