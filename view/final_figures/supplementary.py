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

    concentration_factor = 0.0476 / 526

    fig, axs = plt.subplots(
        2, 2, facecolor='white',
        figsize=(7.5, 5), tight_layout=True)

    axs = axs.flatten()

    for cg_id, ax in zip(CROSSING_GROUP_IDS, axs):

        cg = session.query(models.CrossingGroup).filter_by(id=cg_id).first()

        disc_ths = session.query(models.DiscriminationThreshold).\
            filter_by(crossing_group=cg, variable=RESPONSE_VAR)

        ths = np.array([disc_th.odor_threshold for disc_th in disc_ths])

        if 'fly' in cg_id:

            ths *= concentration_factor

        means = np.array([disc_th.time_avg_difference for disc_th in disc_ths], dtype=float)
        lbs = np.array([disc_th.lower_bound for disc_th in disc_ths], dtype=float)
        ubs = np.array([disc_th.upper_bound for disc_th in disc_ths], dtype=float)

        ax.plot(ths, means, color='k', lw=3)
        ax.fill_between(ths, lbs, ubs, color='k', alpha=.3)

        ax.set_xlim(CROSSING_GROUP_X_LIMS[cg.id])

        if 'fly' in cg_id:

            ax.set_xlabel('threshold (% ethanol)')

        else:

            ax.set_xlabel('threshold (ppm CO2)')

        for xtl in ax.get_xticklabels():

            xtl.set_rotation(60)

        ax.set_ylabel('mean heading diff.\nbetween groups')
        ax.set_title(CROSSING_GROUP_LABELS[cg.id])

    for ax in axs:

        set_fontsize(ax, FONT_SIZE)

    return fig


def crossing_number_distributions(
        CROSSING_GROUPS, AX_GRID):
    """
    Plot histograms of the number of odor crossings per trajectory.
    """

    crossing_numbers_all = {}

    for cg_id in CROSSING_GROUPS:

        # get crossing group and trajectories

        cg = session.query(models.CrossingGroup).get(cg_id)
        expt = cg.experiment

        trajs = session.query(models.Trajectory).filter_by(
            experiment=expt, odor_state='on', clean=True).all()

        crossing_numbers = []

        for traj in trajs:

            crossing_numbers.append(len(
                session.query(models.Crossing).filter_by(
                    crossing_group=cg, trajectory=traj).all()))

        crossing_numbers_all[cg_id] = crossing_numbers

    # MAKE PLOTS

    fig_size = (6 * AX_GRID[1], 3 * AX_GRID[0])

    fig, axs = plt.subplots(*AX_GRID,
        figsize=fig_size, sharex=True, sharey=True, tight_layout=True)

    for ax, cg_id in zip(axs.flatten(), CROSSING_GROUPS):

        bins = np.arange(-1, np.max(crossing_numbers_all[cg_id])) + 0.5

        ax.hist(crossing_numbers_all[cg_id], bins=bins, lw=0, normed=True)

        ax.set_xlabel('number of crossings')
        ax.set_ylabel('proportion\nof trajectories')
        ax.set_title('{}...'.format(cg_id[:15]))

    for ax in axs.flatten():

        set_fontsize(ax, 16)

    return fig


def trajectory_duration_distributions(
        EXPERIMENTS, AX_GRID):
    """
    Plot histogram of trajectory lengths.
    """

    traj_durations = {}

    for expt_id in EXPERIMENTS:

        trajs = session.query(models.Trajectory).filter_by(
            experiment_id=expt_id, odor_state='on', clean=True).all()

        traj_durations[expt_id] = [traj.duration for traj in trajs]

    fig_size = (6 * AX_GRID[1], 3 * AX_GRID[0])

    fig, axs = plt.subplots(*AX_GRID,
        figsize=fig_size, sharex=True, sharey=True, tight_layout=True)

    for ax, expt_id in zip(axs.flatten(), EXPERIMENTS):

        ax.hist(traj_durations[expt_id], bins=50, lw=0, normed=True)

        ax.set_xlabel('duration (s)')
        ax.set_ylabel('proportion\nof trajectories')
        ax.set_title(expt_id)

    for ax in axs.flatten():

        set_fontsize(ax, 16)

    return fig


def bias_vs_n_crossings(NS, SQRT_K_0, SQRT_K_S, BIAS):
    """
    Show how the bias and certainty vary with number of plume crossings,
    given the prior and source covariances.
    """

    k_0 = np.array([
        [SQRT_K_0**2, 0],
        [0, SQRT_K_0**2],
    ])

    k_s = np.array([
        [SQRT_K_S**2, 0],
        [0, SQRT_K_S**2],
    ])

    def certainty(n, k_0, k_s):

        k_inv = np.linalg.inv(k_0) + n*np.linalg.inv(k_s)

        return np.linalg.det(k_inv)

    def bias_uw(n, k_0, k_s):

        c = certainty(n, k_0, k_s)
        bias = np.array([c, 1])
        bias *= (BIAS / np.linalg.norm(bias))

        return bias[0]

    cs = [certainty(n, k_0, k_s) for n in NS]
    bias_uws = [bias_uw(n, k_0, k_s) for n in NS]

    fig, ax = plt.subplots(1, 1, figsize=(5, 4), tight_layout=True)

    ax_twin = ax.twinx()

    ax.plot(NS, bias_uws, lw=2, color='r')
    ax.axhline(BIAS, lw=2, color='k')

    ax_twin.plot(NS, cs, lw=2, color='b')

    ax.set_xlabel('number of crossings')
    ax.set_ylabel('upwind bias', color='r')
    ax_twin.set_ylabel('certainty (1/det(K))', color='b', fontsize=14)

    set_fontsize(ax, 14)

    return fig

