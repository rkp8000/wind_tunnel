"""
Plot trajectories from the raw database and possibly save them in sample database.
"""
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from connect import session
import models


EXPERIMENT_ID = 'fruitfly_0.3mps_checkerboard_floor'
ODOR_STATES = ('on', 'none', 'afterodor')
WALL_BOUNDS = ((-0.3, 1.), (-0.15, 0.15), (-0.15, 0.15))

FACE_COLOR = 'white'
FIG_SIZE = (10, 16)
PLOT_ODOR_THRESHOLD = 10


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

# loop through odor states
for odor_state in ODOR_STATES:
    print('Current odor state: {}'.format(odor_state))
    sample_group = '_'.join([EXPERIMENT_ID, 'odor', odor_state])

    trajs = session.query(models.Trajectory).filter_by(experiment_id=EXPERIMENT_ID, odor_state=odor_state)

    # loop through trajectories
    for traj in trajs:

        # get relevant timepoint quantities
        positions = traj.positions(session)
        dist = dist_from_wall(positions)
        spd = speed(traj.velocities(session))
        odors = traj.odors(session)
        t = np.arange(len(odors)) / 100

        # plot positions
        colors = (odors > PLOT_ODOR_THRESHOLD).astype(int)
        colors[0] = 2
        axs[0].scatter(positions[:, 0], positions[:, 1], c=colors, lw=0, cmap=cm.hot)
        axs[0].set_ylabel('y')
        axs[1].scatter(positions[:, 0], positions[:, 2], c=colors, lw=0, cmap=cm.hot)
        axs[1].set_ylabel('z')
        axs[1].set_xlabel('x')

        # plot odor
        axs[2].plot(t, odors)
        axs[2].set_xlabel('t')
        axs[2].set_ylabel('odor')

        # plot speed and distance to wall overlaid
        axs[3].plot(t, speed, c='b')
        axs[3].set_xlabel('t')
        axs[3].set_ylabel('speed', color='b')

        axs[3].twin = axs[3].twinx()
        axs[3].twin.plot(t, dist, c='r')
        axs[3].twin.set_ylabel('dist from wall', color='r')

        plt.draw()

        command = raw_input('command? [s=save, n=next odor state]')

        if command == 's':
            sample_trajectory = models.SampleTrajectory(sample_group=sample_group,
                                                        experiment_id=EXPERIMENT_ID,
                                                        odor_state=odor_state,
                                                        trajectory=traj)
            session.add(sample_trajectory)
            session.commit()
            print('Trajectory "{}" saved in sample database.'.format(traj.id))
        elif command == 'n':
            print('Moving on to next odor state...')
            break