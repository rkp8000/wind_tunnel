"""
Loop through visualizations of distributions of kinematic quantities.
"""
from __future__ import print_function, division
import pickle
import matplotlib.pyplot as plt
from db_api.connect import session
from db_api import models
plt.ion()

ODOR_STATES = ['on', 'none', 'afterodor']
QUANTITIES = ['odor',
              'position_x',
              'position_y',
              'position_z',
              'velocity_x',
              'velocity_y',
              'velocity_z',
              'velocity_a',
              'acceleration_x',
              'acceleration_y',
              'acceleration_z',
              'acceleration_a',
              'heading_xy',
              'heading_xz',
              'heading_xyz',
              'angular_velocity_x',
              'angular_velocity_y',
              'angular_velocity_z',
              'angular_velocity_a',
              'angular_acceleration_x',
              'angular_acceleration_y',
              'angular_acceleration_z',
              'angular_acceleration_a',
              'distance_from_wall']

FACE_COLOR = 'white'
AX_SIZE = (8, 4)
DATA_COLORS = ('b', 'g', 'r')
LW = 2

expts = list(session.query(models.Experiment))
fig_size = (AX_SIZE[0], AX_SIZE[1] * len(expts))
fig, axs = plt.subplots(len(expts), 1, facecolor=FACE_COLOR, figsize=fig_size, tight_layout=True)

for quantity in QUANTITIES:
    print('Showing distributions of "{}"'.format(quantity))

    [ax.cla() for ax in axs]

    for ax, expt in zip(axs, expts):
        for color, odor_state in zip(DATA_COLORS, ODOR_STATES):

            # get the distribution meta data from the database
            tpd = session.query(models.TimepointDistribution).\
                filter_by(variable=quantity, experiment=expt, odor_state=odor_state).first()

            ax.plot(tpd.bincs, tpd.cts, color=color, lw=LW)

            ax.set_title(expt.id)
            ax.set_ylabel('counts (normed)')

    axs[-1].set_xlabel(quantity)
    axs[0].legend(ODOR_STATES)
    plt.draw()
    raw_input('Press enter to continue:')