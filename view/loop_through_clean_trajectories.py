"""
Plot clean trajectories from analysis database.
"""
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from db_api.connect import session
from db_api import models


EXPERIMENT_ID = 'fruitfly_0.4mps_checkerboard_floor'
ODOR_STATES = ('on', 'none', 'afterodor')
WALL_BOUNDS = ((-0.3, 1.), (-0.15, 0.15), (-0.15, 0.15))

FACE_COLOR = 'white'
FIG_SIZE = (10, 12)

if 'mosquito' in EXPERIMENT_ID:
    PLOT_ODOR_THRESHOLD = 450
    ODOR_YLIM = (400, 1000)
else:
    PLOT_ODOR_THRESHOLD = 10
    ODOR_YLIM = (0, 550)


def speed(velocities):
    """Return a speed time-series given a velocity time-series."""
    return np.linalg.norm(velocities, axis=1)


def dist_from_wall(positions, wall_bounds=WALL_BOUNDS):
    """Return the distance from the nearest wall."""
    above_x = positions[:, 0] - wall_bounds[0][0]
    below_x = wall_bounds[0][1] - positions[:, 0]
    above_y = positions[:, 1] - wall_bounds[1][0]
    below_y = wall_bounds[1][1] - positions[:, 1]
    above_z = positions[:, 2] - wall_bounds[2][0]
    below_z = wall_bounds[2][1] - positions[:, 2]

    dist_all_walls = np.array([above_x, below_x,
                               above_y, below_y,
                               above_z, below_z])

    return np.min(dist_all_walls, axis=0)


plt.ion()
fig, axs = plt.subplots(4, 1, facecolor=FACE_COLOR, figsize=FIG_SIZE, tight_layout=True)
axs[3].twin = axs[3].twinx()

# loop through odor states
for odor_state in ODOR_STATES:
    print('Current odor state: {}'.format(odor_state))
    sample_group = '_'.join([EXPERIMENT_ID, 'odor', odor_state])

    trajs = session.query(models.Trajectory).filter_by(experiment_id=EXPERIMENT_ID,
                                                       odor_state=odor_state,
                                                       clean=True).all()

    # loop through trajectories
    t_ctr = 0
    n_trajs = len(trajs)

    if trajs:
        keep_going = True
    else:
        keep_going = False

    while keep_going:

        t_ctr %= n_trajs

        traj = trajs[t_ctr]
        # get relevant timepoint quantities
        positions = traj.positions(session)
        dist = dist_from_wall(positions)
        spd = speed(traj.velocities(session))
        odors = traj.odors(session)
        t = np.arange(len(odors)) / 100

        [ax.cla() for ax in axs]
        axs[3].twin.cla()

        # plot positions
        colors = (odors > PLOT_ODOR_THRESHOLD).astype(int)
        colors[0] = 2
        axs[0].scatter(positions[:, 0], positions[:, 1], c=colors, lw=0, cmap=cm.hot)
        axs[0].set_ylabel('y (m)')
        axs[0].set_title('Trajectory {}'.format(t_ctr))
        axs[1].scatter(positions[:, 0], positions[:, 2], c=colors, lw=0, cmap=cm.hot)
        axs[1].set_ylabel('z (m)')
        axs[1].set_xlabel('x (m)')

        # plot odor
        axs[2].plot(t, odors, lw=2)
        axs[2].set_ylim(ODOR_YLIM)
        axs[2].set_xlabel('t (s)')
        axs[2].set_ylabel('odor (a.u.)')

        # plot speed and distance to wall overlaid
        axs[3].plot(t, spd, c='b', lw=2)
        axs[3].set_ylim(0, 0.6)
        axs[3].set_xlabel('t (s)')
        axs[3].set_ylabel('speed (m/s)', color='b')
        [tl.set_color('b') for tl in axs[3].get_yticklabels()]

        axs[3].twin.plot(t, dist, c='r', lw=2)
        axs[3].twin.set_ylabel('dist from wall (m)', color='r')
        [tl.set_color('r') for tl in axs[3].twin.get_yticklabels()]

        plt.draw()

        command = raw_input('command [n=next, b=back, q=quit]:')

        if command == 'b':
            t_ctr -= 1
        elif command == 'q':
            keep_going = False
            plt.show(block=True)
        else:
            t_ctr += 1