"""
Functions for calculating kinematic quantities of flight trajectories.
"""
from __future__ import print_function, division
import numpy as np
np.seterr(all='ignore')


def norm(vectors, shape='1darray'):
    """
    Calculate the norm, making sure to return a float.

    :param vectors: 2D array of vectors
    :param shape: whether to return '1darray' or '2darray' w/ same shape as vectors
    :return: vector norm
    """

    n = np.linalg.norm(vectors.astype(float), axis=1)

    if shape == '1darray':
        return n
    elif shape == '2darray':
        return np.tile(n, (vectors.shape[1], 1)).T


def acceleration(velocities, dt):
    """
    Calculate acceleration from velocity using central differences.

    :param velocities: 2D array of velocities (rows are timepoints)
    :param dt: interval between timesteps
    :return: array of accelerations
    """

    return np.transpose([np.gradient(velocities[:, dim], dt) for dim in range(velocities.shape[1])])


def heading(velocities):
    """
    Calculate heading in xy and xz plane, as well as 3d heading.

    :param velocities: 2D array of velocities (rows are timepoints)
    :return: array of headings (first col xy, second col xz, third col xyz)
    """

    v_xy = velocities[:, [0, 1]].copy().astype(float)
    v_xz = velocities[:, [0, 2]].copy().astype(float)
    v_xyz = velocities.copy().astype(float)

    # normalize each set of velocities
    norm_xy = norm(v_xy)
    norm_xz = norm(v_xz)
    norm_xyz = norm(v_xyz)

    v_xy /= np.tile(norm_xy, (2, 1)).T
    v_xz /= np.tile(norm_xz, (2, 1)).T
    v_xyz /= np.tile(norm_xyz, (3, 1)).T

    # array of upwind vectors
    uw_vec = np.transpose([-np.ones((len(velocities),), dtype=float),
                           np.zeros((len(velocities),), dtype=float),
                           np.zeros((len(velocities),), dtype=float)])

    heading_xy = np.arccos((uw_vec[:, [0, 1]] * v_xy).sum(axis=1))
    heading_xz = np.arccos((uw_vec[:, [0, 2]] * v_xz).sum(axis=1))
    heading_xyz = np.arccos((uw_vec * v_xyz).sum(axis=1))

    heading_xy[norm_xy == 0] = 0
    heading_xz[norm_xz == 0] = 0
    heading_xyz[norm_xyz == 0] = 0

    return np.transpose([heading_xy, heading_xz, heading_xyz]) * 180 / np.pi


def angular_velocity(velocities, dt):
    """
    Calculate angular velocities.

    :param velocities: 2D array of velocities (rows are timepoints)
    :param dt: interval between timesteps
    :return: array of angular velocities
    """

    # calculate normalized velocity vector
    v_norm = velocities / norm(velocities, shape='2darray')

    # get angle between each consecutive pair of normalized velocity vectors
    d_theta = np.arccos((v_norm[:-1, :] * v_norm[1:, :]).sum(1))
    a_vel_mag = d_theta / dt
    # calculate the direction of angular change by computing the cross-
    # product between each consecutive pair of normalized velocity vectors
    cp = np.cross(v_norm[:-1, :], v_norm[1:, :])
    # normalize the cross product array
    cp /= np.tile(np.linalg.norm(cp.astype(float), axis=1), (3, 1)).T
    # create angular velocity array and set to zero the places where the magnitude is zero
    a_vel = cp * np.tile(a_vel_mag, (3, 1)).T
    a_vel[a_vel_mag == 0] = 0
    # correct size so that it matches the size of the velocity array
    a_vel_full = np.zeros((a_vel.shape[0] + 1, a_vel.shape[1]), dtype=float)
    a_vel_full[:-1] += a_vel
    a_vel_full[1:] += a_vel
    a_vel_full[1:-1] /= 2.

    return a_vel_full


def distance_from_wall(positions, wall_bounds):
    """
    Calculate distance from nearest wall.

    :param positions: 2D array of positions (rows are timepoints)
    :param wall_bounds: wall boundaries ((x_lower, x_upper), (y_lower, ...), ...)
    :return: 1D array of distances from wall
    """

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