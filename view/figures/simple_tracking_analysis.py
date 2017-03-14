from __future__ import division, print_function
from itertools import product as cproduct
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import optimize, stats

from db_api import models
from db_api.connect import session

from kinematics import heading, angular_velocity
from plot import set_font_size
from simple_tracking import GaussianLaminarPlume
from simple_tracking import CenterlineInferringAgent, SurgingAgent
from stats import nansem
from time_series import segment_by_threshold


def example_trajectory_centerline_no_plume(SEED, DURATION, DT, TAU, NOISE, BIAS, BOUNDS):
    """
    Create an example trajectory and plot some of the resulting covariates.
    """

    # build plume and agent

    pl = GaussianLaminarPlume(0, np.array([0., 0]), [1., 1.])
    ag = CenterlineInferringAgent(
        tau=TAU, noise=NOISE, bias=BIAS, threshold=np.inf,
        hit_trigger='peak', hit_influence=0,
        k_0=np.eye(2), k_s=np.eye(2), tau_memory=1, bounds=BOUNDS)

    # generate the trajectory

    np.random.seed(SEED)

    start_pos = np.array([
        np.random.uniform(*BOUNDS[0]),
        np.random.uniform(*BOUNDS[1]),
        np.random.uniform(*BOUNDS[2]),
    ])

    traj = ag.track(pl, start_pos, DURATION, DT)

    # plot trajectory

    fig = plt.figure(figsize=(15, 10), tight_layout=True)
    axs = []

    axs.append(fig.add_subplot(2, 1, 1))

    axs[0].plot(traj['xs'][:, 0], traj['xs'][:, 1], lw=2, color='k', zorder=0)
    axs[0].scatter(traj['xs'][0, 0], traj['xs'][0, 1], lw=0, c='r', zorder=1, s=100)

    axs[0].set_xlim(*BOUNDS[0])
    axs[0].set_ylim(*BOUNDS[1])

    axs[0].set_xlabel('x (m)')
    axs[0].set_ylabel('y (m)')

    axs[0].set_title('example trajectory')

    # plot some histograms

    speeds = np.linalg.norm(traj['vs'], axis=1)
    ws = np.linalg.norm(angular_velocity(traj['vs'], DT), axis=1)
    ws = ws[~np.isnan(ws)]

    axs.append(fig.add_subplot(2, 2, 3))
    axs.append(fig.add_subplot(2, 2, 4))

    axs[1].hist(speeds, bins=30, lw=0, normed=True)
    axs[2].hist(ws, bins=30, lw=0, normed=True)

    axs[1].set_xlabel('speed (m/s)')
    axs[2].set_xlabel('ang. vel (rad/s)')

    axs[1].set_ylabel('relative counts')

    for ax in axs:

        set_font_size(ax, 16)

    return fig


def example_trajectory_centerline_with_plume(
        SEED, DURATION, DT, TAU, NOISE, BIAS, THRESHOLD, HIT_INFLUENCE,
        TAU_MEMORY, K_0, K_S, BOUNDS,
        PL_CONC, PL_MEAN, PL_STD):
    """
    Create an example trajectory and plot some of the resulting covariates.
    """

    # build plume and agent

    pl = GaussianLaminarPlume(PL_CONC, PL_MEAN, PL_STD)

    k_0 = K_0 * np.eye(2)
    k_s = K_S * np.eye(2)

    ag = CenterlineInferringAgent(
        tau=TAU, noise=NOISE, bias=BIAS, threshold=THRESHOLD,
        hit_trigger='peak', hit_influence=HIT_INFLUENCE,
        k_0=k_0, k_s=k_s, tau_memory=TAU_MEMORY, bounds=BOUNDS)

    # generate the trajectory

    np.random.seed(SEED)

    start_pos = np.array([
        np.random.uniform(*BOUNDS[0]),
        np.random.uniform(*BOUNDS[1]),
        np.random.uniform(*BOUNDS[2]),
    ])

    traj = ag.track(pl, start_pos, DURATION, DT)

    # plot trajectory

    fig = plt.figure(figsize=(15, 10), tight_layout=True)
    axs = [fig.add_subplot(4, 1, 1)]

    axs[-1].plot(traj['xs'][:, 0], traj['xs'][:, 1], lw=2, color='k', zorder=0)
    axs[-1].scatter(traj['xs'][0, 0], traj['xs'][0, 1], lw=0, c='r', zorder=1, s=100)

    axs[-1].set_xlim(*BOUNDS[0])
    axs[-1].set_ylim(*BOUNDS[1])

    axs[-1].set_xlabel('x (m)')
    axs[-1].set_ylabel('y (m)')

    axs[-1].set_title('example trajectory')

    # plot some histograms

    speeds = np.linalg.norm(traj['vs'], axis=1)
    ws = np.linalg.norm(angular_velocity(traj['vs'], DT), axis=1)
    ws = ws[~np.isnan(ws)]

    axs.append(fig.add_subplot(4, 2, 3))
    axs.append(fig.add_subplot(4, 2, 4))

    axs[-2].hist(speeds, bins=30, lw=0, normed=True)
    axs[-1].hist(ws, bins=30, lw=0, normed=True)

    axs[-2].set_xlabel('speed (m/s)')
    axs[-1].set_xlabel('ang. vel (rad/s)')

    axs[-2].set_ylabel('relative counts')

    axs.append(fig.add_subplot(4, 1, 3))
    axs.append(axs[-1].twinx())

    ts = traj['ts']
    odors = traj['odors']
    cl_vars = np.trace(traj['centerline_ks'], axis1=1, axis2=2)

    axs[-2].plot(ts, odors, color='r', lw=2)
    axs[-1].plot(ts, cl_vars, color='k', lw=2)

    axs[-2].set_xlabel('time (s)')
    axs[-2].set_ylabel('odor')
    axs[-1].set_ylabel('centerline var')

    axs.append(fig.add_subplot(4, 1, 4))
    axs.append(axs[-1].twinx())

    bs = traj['bs']

    axs[-2].plot(ts, odors, color='r', lw=2)
    axs[-1].plot(ts, bs[:, 0], color='k', lw=2)

    axs[-2].set_xlabel('time (s)')
    axs[-2].set_ylabel('odor')
    axs[-1].set_ylabel('upwind bias')

    for ax in axs:

        set_font_size(ax, 16)

    return fig


def example_trajectory_surging_no_plume(
        SEED, DURATION, DT, TAU, NOISE, BIAS, BOUNDS):
    """
    Create an example trajectory and plot some of the resulting covariates.
    """

    # build plume and agent

    pl = GaussianLaminarPlume(0, np.array([0., 0]), [1., 1.])
    ag = SurgingAgent(
        tau=TAU, noise=NOISE, bias=BIAS, threshold=np.inf,
        hit_trigger='peak', surge_strength=0, tau_surge=0, bounds=BOUNDS)

    # generate the trajectory

    np.random.seed(SEED)

    start_pos = np.array([
        np.random.uniform(*BOUNDS[0]),
        np.random.uniform(*BOUNDS[1]),
        np.random.uniform(*BOUNDS[2]),
    ])

    traj = ag.track(pl, start_pos, DURATION, DT)

    # plot trajectory

    fig = plt.figure(figsize=(15, 10), tight_layout=True)
    axs = []

    axs.append(fig.add_subplot(2, 1, 1))

    axs[0].plot(traj['xs'][:, 0], traj['xs'][:, 1], lw=2, color='k', zorder=0)
    axs[0].scatter(traj['xs'][0, 0], traj['xs'][0, 1], lw=0, c='r', zorder=1, s=100)

    axs[0].set_xlim(*BOUNDS[0])
    axs[0].set_ylim(*BOUNDS[1])

    axs[0].set_xlabel('x (m)')
    axs[0].set_ylabel('y (m)')

    axs[0].set_title('example trajectory')

    # plot some histograms

    speeds = np.linalg.norm(traj['vs'], axis=1)
    ws = np.linalg.norm(angular_velocity(traj['vs'], DT), axis=1)
    ws = ws[~np.isnan(ws)]

    axs.append(fig.add_subplot(2, 2, 3))
    axs.append(fig.add_subplot(2, 2, 4))

    axs[1].hist(speeds, bins=30, lw=0, normed=True)
    axs[2].hist(ws, bins=30, lw=0, normed=True)

    axs[1].set_xlabel('speed (m/s)')
    axs[2].set_xlabel('ang. vel (rad/s)')

    axs[1].set_ylabel('relative counts')

    for ax in axs:

        set_font_size(ax, 16)

    return fig


def example_trajectory_surging_with_plume(
        SEED, DURATION, DT, TAU, NOISE, BIAS, THRESHOLD,
        SURGE_AMP, TAU_SURGE, BOUNDS,
        PL_CONC, PL_MEAN, PL_STD):
    """
    Create an example trajectory and plot some of the resulting covariates.
    """

    # build plume and agent

    pl = GaussianLaminarPlume(PL_CONC, PL_MEAN, PL_STD)

    ag = SurgingAgent(
        tau=TAU, noise=NOISE, bias=BIAS, threshold=THRESHOLD, hit_trigger='peak',
        surge_amp=SURGE_AMP, tau_surge=TAU_SURGE, bounds=BOUNDS)

    # generate the trajectory

    np.random.seed(SEED)

    start_pos = np.array([
        np.random.uniform(*BOUNDS[0]),
        np.random.uniform(*BOUNDS[1]),
        np.random.uniform(*BOUNDS[2]),
    ])

    traj = ag.track(pl, start_pos, DURATION, DT)

    # plot trajectory

    fig = plt.figure(figsize=(15, 10), tight_layout=True)
    axs = [fig.add_subplot(3, 1, 1)]

    axs[-1].plot(traj['xs'][:, 0], traj['xs'][:, 1], lw=2, color='k', zorder=0)
    axs[-1].scatter(traj['xs'][0, 0], traj['xs'][0, 1], lw=0, c='r', zorder=1, s=100)

    axs[-1].set_xlim(*BOUNDS[0])
    axs[-1].set_ylim(*BOUNDS[1])

    axs[-1].set_xlabel('x (m)')
    axs[-1].set_ylabel('y (m)')

    axs[-1].set_title('example trajectory')

    # plot some histograms

    speeds = np.linalg.norm(traj['vs'], axis=1)
    ws = np.linalg.norm(angular_velocity(traj['vs'], DT), axis=1)
    ws = ws[~np.isnan(ws)]

    axs.append(fig.add_subplot(3, 2, 3))
    axs.append(fig.add_subplot(3, 2, 4))

    axs[-2].hist(speeds, bins=30, lw=0, normed=True)
    axs[-1].hist(ws, bins=30, lw=0, normed=True)

    axs[-2].set_xlabel('speed (m/s)')
    axs[-1].set_xlabel('ang. vel (rad/s)')

    axs[-2].set_ylabel('relative counts')

    axs.append(fig.add_subplot(3, 1, 3))
    axs.append(axs[-1].twinx())

    ts = traj['ts']
    odors = traj['odors']
    surge_forces = traj['surges']

    axs[-2].plot(ts, odors, color='r', lw=2)
    axs[-1].plot(ts, surge_forces, color='k', lw=2)

    axs[-2].set_xlabel('time (s)')
    axs[-2].set_ylabel('odor', color='red')
    axs[-1].set_ylabel('surge forces')

    for ax in axs:

        set_font_size(ax, 16)

    return fig



def optimize_model_params(
        SEED,
        DURATION, DT, BOUNDS,
        EXPERIMENT, ODOR_STATE,
        MAX_TRAJS_EMPIRICAL,
        N_TIME_POINTS_EMPIRICAL,
        SAVE_FILE_PREFIX,
        INITIAL_PARAMS, MAX_ITERS):
    """
    Find optimal model parameters by fitting speed and angular velocity distributions of empirical
    data.
    """

    # check to see if empirical time points have already been saved

    file_name = '{}_{}_odor_{}.npy'.format(SAVE_FILE_PREFIX, EXPERIMENT, ODOR_STATE)

    if os.path.isfile(file_name):

        empirical = np.load(file_name)[0]

    else:

        print('extracting time points from data')

        # get all trajectories

        trajs = session.query(models.Trajectory).filter_by(
            experiment_id=EXPERIMENT, odor_state=ODOR_STATE, clean=True).\
            limit(MAX_TRAJS_EMPIRICAL).all()

        # get all speeds and angular velocities

        cc = np.concatenate
        speeds_empirical = cc([traj.velocities_a(session) for traj in trajs])
        ws_empirical = cc([traj.angular_velocities_a(session) for traj in trajs])
        ys_empirical = cc([traj.timepoint_field(session, 'position_y') for traj in trajs])

        # sample a set of speeds and ws

        np.random.seed(SEED)

        speeds_empirical = np.random.choice(speeds_empirical, N_TIME_POINTS_EMPIRICAL, replace=False)
        ws_empirical = np.random.choice(ws_empirical, N_TIME_POINTS_EMPIRICAL, replace=False)
        ys_empirical = np.random.choice(ys_empirical, N_TIME_POINTS_EMPIRICAL, replace=False)

        empirical = {'speeds': speeds_empirical, 'ws': ws_empirical, 'ys': ys_empirical}

        # save them for easy access next time

        np.save(file_name, np.array([empirical]))

    print('performing optimization')

    # make a plume

    pl = GaussianLaminarPlume(0, np.zeros((2,)), np.ones((2,)))

    # define function to be optimized

    def optim_fun(p):

        np.random.seed(SEED)

        start_pos = np.array([
            np.random.uniform(*BOUNDS[0]),
            np.random.uniform(*BOUNDS[1]),
            np.random.uniform(*BOUNDS[2]),
        ])

        # make agent and trajectory

        ag = CenterlineInferringAgent(
            tau=p[0], noise=p[1], bias=p[2], threshold=np.inf,
            hit_trigger='peak', hit_influence=0,
            k_0=np.eye(2), k_s=np.eye(2), tau_memory=1, bounds=BOUNDS)

        traj = ag.track(pl, start_pos, DURATION, DT)

        speeds = np.linalg.norm(traj['vs'], axis=1)
        ws = np.linalg.norm(angular_velocity(traj['vs'], DT), axis=1)
        ws = ws[~np.isnan(ws)]
        ys = traj['xs'][:, 1]

        ks_speeds = stats.ks_2samp(speeds, empirical['speeds'])[0]
        ks_ws = stats.ks_2samp(ws, empirical['ws'])[0]
        ks_ys = stats.ks_2samp(ys, empirical['ys'])[0]

        val = ks_speeds + ks_ws + ks_ys

        # punish unallowable values

        if np.any(p < 0):

            val += 10000

        return val

    # optimize it

    p_best = optimize.fmin(optim_fun, np.array(INITIAL_PARAMS), maxiter=MAX_ITERS)

    # generate one final trajectory

    np.random.seed(SEED)

    start_pos = np.array([
        np.random.uniform(*BOUNDS[0]),
        np.random.uniform(*BOUNDS[1]),
        np.random.uniform(*BOUNDS[2]),
    ])

    ag = CenterlineInferringAgent(
        tau=p_best[0], noise=p_best[1], bias=p_best[2], threshold=np.inf,
        hit_trigger='peak', hit_influence=0,
        k_0=np.eye(2), k_s=np.eye(2), tau_memory=1, bounds=BOUNDS)

    traj = ag.track(pl, start_pos, DURATION, DT)

    speeds = np.linalg.norm(traj['vs'], axis=1)
    ws = np.linalg.norm(angular_velocity(traj['vs'], DT), axis=1)
    ws = ws[~np.isnan(ws)]
    ys = traj['xs'][:, 1]

    # make plots of things that have been optimized

    ## get bins

    speed_max = max(speeds.max(), empirical['speeds'].max())
    bins_speed = np.linspace(0, speed_max, 41, endpoint=True)
    bincs_speed = 0.5 * (bins_speed[:-1] + bins_speed[1:])

    w_max = max(ws.max(), empirical['ws'].max())
    bins_w = np.linspace(0, w_max, 41, endpoint=True)
    bincs_w = 0.5 * (bins_w[:-1] + bins_w[1:])

    bins_y = np.linspace(BOUNDS[1][0], BOUNDS[1][1], 41, endpoint=True)
    bincs_y = 0.5 * (bins_y[:-1] + bins_y[1:])

    cts_speed, _ = np.histogram(speeds, bins=bins_speed, normed=True)
    cts_speed_empirical, _ = np.histogram(empirical['speeds'], bins=bins_speed, normed=True)

    cts_w, _ = np.histogram(ws, bins=bins_w, normed=True)
    cts_w_empirical, _ = np.histogram(empirical['ws'], bins=bins_w, normed=True)

    cts_y, _ = np.histogram(ys, bins=bins_y, normed=True)
    cts_y_empirical, _ = np.histogram(empirical['ys'], bins=bins_y, normed=True)

    fig = plt.figure(figsize=(15, 8), tight_layout=True)
    axs = []

    axs.append(fig.add_subplot(2, 3, 1))
    axs.append(fig.add_subplot(2, 3, 2))
    axs.append(fig.add_subplot(2, 3, 3))

    axs[0].plot(bincs_speed, cts_speed_empirical, lw=2, color='k')
    axs[0].plot(bincs_speed, cts_speed, lw=2, color='r')

    axs[0].set_xlabel('speed (m/s)')
    axs[0].set_ylabel('rel. counts')

    axs[0].legend(['data', 'model'], fontsize=16)

    axs[1].plot(bincs_w, cts_w_empirical, lw=2, color='k')
    axs[1].plot(bincs_w, cts_w, lw=2, color='r')

    axs[1].set_xlabel('ang. vel. (rad/s)')

    axs[2].plot(bincs_y, cts_y_empirical, lw=2, color='k')
    axs[2].plot(bincs_y, cts_y, lw=2, color='r')

    axs[2].set_xlabel('y (m)')

    axs.append(fig.add_subplot(2, 1, 2))

    axs[3].plot(traj['xs'][:500, 0], traj['xs'][:500, 1], lw=2, color='k', zorder=0)
    axs[3].scatter(traj['xs'][0, 0], traj['xs'][0, 1], lw=0, c='r', zorder=1, s=100)

    axs[3].set_xlim(*BOUNDS[0])
    axs[3].set_ylim(*BOUNDS[1])

    axs[3].set_xlabel('x (m)')
    axs[3].set_ylabel('y (m)')

    axs[3].set_title('example trajectory')

    for ax in axs:

        set_font_size(ax, 16)

    # print out parameters

    print('best params:')
    print('tau = {}'.format(p_best[0]))
    print('noise = {}'.format(p_best[1]))
    print('bias = {}'.format(p_best[2]))

    return fig


def crossing_triggered_headings_all(
        SEED,
        N_TRAJS, DURATION, DT,
        TAU, NOISE, BIAS, AGENT_THRESHOLD,
        HIT_INFLUENCE, TAU_MEMORY, K_0, K_S,
        BOUNDS,
        PL_CONC, PL_MEAN, PL_STD,
        ANALYSIS_THRESHOLD, H_MIN_PEAK, H_MAX_PEAK,
        SUBTRACT_PEAK_HEADING, T_BEFORE, T_AFTER, Y_LIM):
    """
    Fly several agents through a simulated plume and plot their plume-crossing-triggered
    headings.
    """

    # build plume and agent

    pl = GaussianLaminarPlume(PL_CONC, PL_MEAN, PL_STD)

    k_0 = K_0 * np.eye(2)
    k_s = K_S * np.eye(2)

    ag = CenterlineInferringAgent(
        tau=TAU, noise=NOISE, bias=BIAS, threshold=AGENT_THRESHOLD,
        hit_trigger='peak', hit_influence=HIT_INFLUENCE,
        k_0=k_0, k_s=k_s, tau_memory=TAU_MEMORY, bounds=BOUNDS)

    # generate trajectories

    np.random.seed(SEED)

    trajs = []

    for _ in range(N_TRAJS):

        # choose random start position

        start_pos = np.array([
            np.random.uniform(*BOUNDS[0]),
            np.random.uniform(*BOUNDS[1]),
            np.random.uniform(*BOUNDS[2]),
        ])

        # make trajectory

        traj = ag.track(plume=pl, start_pos=start_pos, duration=DURATION, dt=DT)

        traj['headings'] = heading(traj['vs'])[:, 2]

        trajs.append(traj)

    crossings = []

    ts_before = int(T_BEFORE / DT)
    ts_after = int(T_AFTER / DT)

    for traj in trajs:

        starts, onsets, peak_times, offsets, ends = \
            segment_by_threshold(traj['odors'], ANALYSIS_THRESHOLD)[0].T

        for start, peak_time, end in zip(starts, peak_times, ends):

            if not (H_MIN_PEAK <= traj['headings'][peak_time] < H_MAX_PEAK):

                continue

            crossing = np.nan * np.zeros((ts_before + ts_after,))

            ts_before_crossing = peak_time - start
            ts_after_crossing = end - peak_time

            if ts_before_crossing >= ts_before:

                crossing[:ts_before] = traj['headings'][peak_time - ts_before:peak_time]

            else:

                crossing[ts_before - ts_before_crossing:ts_before] = \
                    traj['headings'][start:peak_time]

            if ts_after_crossing >= ts_after:

                crossing[ts_before:] = traj['headings'][peak_time:peak_time + ts_after]

            else:

                crossing[ts_before:ts_before + ts_after_crossing] = \
                    traj['headings'][peak_time:end]

            if SUBTRACT_PEAK_HEADING:

                crossing -= crossing[ts_before]

            crossings.append(crossing)

    crossings = np.array(crossings)

    t = np.arange(-ts_before, ts_after) * DT

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)

    h_mean = np.nanmean(crossings, axis=0)
    h_sem = nansem(crossings, axis=0)

    ax.plot(t, crossings.T, lw=0.5, alpha=0.5, color='c', zorder=0)
    ax.plot(t, h_mean, lw=3, color='k')
    ax.fill_between(t, h_mean - h_sem, h_mean + h_sem, color='k', alpha=0.2)

    ax.axvline(0, ls='--', color='gray')

    ax.set_ylim(*Y_LIM)
    ax.set_xlabel('time since peak (s)')

    if SUBTRACT_PEAK_HEADING:

        ax.set_ylabel('change in heading (deg)')

    else:

        ax.set_ylabel('heading (deg)')

    set_font_size(ax, 16)

    return fig


def crossing_triggered_headings_early_late_centerline(
        SEED, N_TRAJS, DURATION, DT,
        TAU, NOISE, BIAS, AGENT_THRESHOLD,
        HIT_INFLUENCE, TAU_MEMORY,
        K_0, K_S, BOUNDS,
        PL_CONC, PL_MEAN, PL_STD,
        ANALYSIS_THRESHOLD,
        H_MIN_PEAK, H_MAX_PEAK,
        X_MIN_PEAK, X_MAX_PEAK,
        EARLY_LESS_THAN,
        SUBTRACT_PEAK_HEADING, T_BEFORE, T_AFTER):
    """
    Fly several agents through a simulated plume and plot their plume-crossing-triggered
    headings.
    """

    # build plume and agent

    pl = GaussianLaminarPlume(PL_CONC, PL_MEAN, PL_STD)

    k_0 = K_0 * np.eye(2)
    k_s = K_S * np.eye(2)

    ag = CenterlineInferringAgent(
        tau=TAU, noise=NOISE, bias=BIAS, threshold=AGENT_THRESHOLD,
        hit_trigger='peak', hit_influence=HIT_INFLUENCE,
        k_0=k_0, k_s=k_s, tau_memory=TAU_MEMORY, bounds=BOUNDS)

    # GENERATE TRAJECTORIES

    np.random.seed(SEED)

    trajs = []

    for _ in range(N_TRAJS):

        # choose random start position

        start_pos = np.array([
            np.random.uniform(*BOUNDS[0]),
            np.random.uniform(*BOUNDS[1]),
            np.random.uniform(*BOUNDS[2]),
        ])

        # make trajectory

        traj = ag.track(plume=pl, start_pos=start_pos, duration=DURATION, dt=DT)

        traj['headings'] = heading(traj['vs'])[:, 2]

        trajs.append(traj)

    # ANALYZE TRAJECTORIES

    n_crossings = []

    # collect early and late crossings

    crossings_early = []
    crossings_late = []

    ts_before = int(T_BEFORE / DT)
    ts_after = int(T_AFTER / DT)

    for traj in trajs:

        starts, onsets, peak_times, offsets, ends = \
            segment_by_threshold(traj['odors'], ANALYSIS_THRESHOLD)[0].T

        n_crossings.append(len(peak_times))

        for ctr, (start, peak_time, end) in enumerate(zip(starts, peak_times, ends)):

            if not (H_MIN_PEAK <= traj['headings'][peak_time] < H_MAX_PEAK):

                continue

            if not (X_MIN_PEAK <= traj['xs'][peak_time, 0] < X_MAX_PEAK):

                continue

            crossing = np.nan * np.zeros((ts_before + ts_after,))

            ts_before_crossing = peak_time - start
            ts_after_crossing = end - peak_time

            if ts_before_crossing >= ts_before:

                crossing[:ts_before] = traj['headings'][peak_time - ts_before:peak_time]

            else:

                crossing[ts_before - ts_before_crossing:ts_before] = \
                    traj['headings'][start:peak_time]

            if ts_after_crossing >= ts_after:

                crossing[ts_before:] = traj['headings'][peak_time:peak_time + ts_after]

            else:

                crossing[ts_before:ts_before + ts_after_crossing] = \
                    traj['headings'][peak_time:end]

            if SUBTRACT_PEAK_HEADING:

                crossing -= crossing[ts_before]

            if ctr < EARLY_LESS_THAN:

                crossings_early.append(crossing)

            else:

                crossings_late.append(crossing)

    n_crossings = np.array(n_crossings)

    crossings_early = np.array(crossings_early)
    crossings_late = np.array(crossings_late)

    t = np.arange(-ts_before, ts_after) * DT

    h_mean_early = np.nanmean(crossings_early, axis=0)
    h_sem_early = nansem(crossings_early, axis=0)

    h_mean_late = np.nanmean(crossings_late, axis=0)
    h_sem_late = nansem(crossings_late, axis=0)

    fig, axs = plt.figure(figsize=(15, 15), tight_layout=True), []

    axs.append(fig.add_subplot(3, 2, 1))
    axs.append(fig.add_subplot(3, 2, 2))

    handles = []

    try:

        handles.append(axs[0].plot(t, h_mean_early, lw=3, color='b', label='early')[0])
        axs[0].fill_between(t, h_mean_early - h_sem_early, h_mean_early + h_sem_early,
            color='b', alpha=0.2)

    except:

        pass

    try:

        handles.append(axs[0].plot(t, h_mean_late, lw=3, color='g', label='late')[0])
        axs[0].fill_between(t, h_mean_late - h_sem_late, h_mean_late + h_sem_late,
            color='g', alpha=0.2)

    except:

        pass

    axs[0].axvline(0, ls='--', color='gray')

    axs[0].set_xlabel('time since peak (s)')

    if SUBTRACT_PEAK_HEADING:

        axs[0].set_ylabel('change in heading (deg)')

    else:

        axs[0].set_ylabel('heading (deg)')

    axs[0].legend(handles=handles, fontsize=16)

    bin_min = -0.5
    bin_max = n_crossings.max() + 0.5

    bins = np.linspace(bin_min, bin_max, bin_max - bin_min + 1, endpoint=True)

    axs[1].hist(n_crossings, bins=bins, lw=0, normed=True)
    axs[1].set_xlim(bin_min, bin_max)

    axs[1].set_xlabel('number of crossings')
    axs[1].set_ylabel('proportion of trajectories')

    axs.append(fig.add_subplot(3, 1, 2))

    axs[2].plot(trajs[0]['xs'][:, 0], trajs[0]['xs'][:, 1])
    axs[2].axhline(0, color='gray', ls='--')

    axs[2].set_xlabel('x (m)')
    axs[2].set_ylabel('y (m)')

    axs.append(fig.add_subplot(3, 1, 3))

    all_xy = np.concatenate([traj['xs'][:, :2] for traj in trajs[:3000]], axis=0)
    x_bins = np.linspace(BOUNDS[0][0], BOUNDS[0][1], 66, endpoint=True)
    y_bins = np.linspace(BOUNDS[1][0], BOUNDS[1][1], 30, endpoint=True)

    axs[3].hist2d(all_xy[:, 0], all_xy[:, 1], bins=(x_bins, y_bins))

    axs[3].set_xlabel('x (m)')
    axs[3].set_ylabel('y (m)')

    for ax in axs:

        set_font_size(ax, 20)

    return fig


def crossing_triggered_headings_early_late_surge(
        SEED, N_TRAJS, DURATION, DT, BOUNDS,
        TAU, NOISE, BIAS, AGENT_THRESHOLD,
        SURGE_AMP, TAU_SURGE,
        PL_CONC, PL_MEAN, PL_STD,
        ANALYSIS_THRESHOLD,
        H_MIN_PEAK, H_MAX_PEAK,
        X_MIN_PEAK, X_MAX_PEAK,
        EARLY_LESS_THAN,
        SUBTRACT_PEAK_HEADING, T_BEFORE, T_AFTER):
    """
    Fly several agents through a simulated plume and plot their plume-crossing-triggered
    headings.
    """

    # build plume and agent

    pl = GaussianLaminarPlume(PL_CONC, PL_MEAN, PL_STD)

    ag = SurgingAgent(
        tau=TAU, noise=NOISE, bias=BIAS, threshold=AGENT_THRESHOLD,
        hit_trigger='peak', surge_amp=SURGE_AMP, tau_surge=TAU_SURGE,
        bounds=BOUNDS)

    # GENERATE TRAJECTORIES

    np.random.seed(SEED)

    trajs = []

    for _ in range(N_TRAJS):

        # choose random start position

        start_pos = np.array([
            np.random.uniform(*BOUNDS[0]),
            np.random.uniform(*BOUNDS[1]),
            np.random.uniform(*BOUNDS[2]),
        ])

        # make trajectory

        traj = ag.track(plume=pl, start_pos=start_pos, duration=DURATION, dt=DT)

        traj['headings'] = heading(traj['vs'])[:, 2]

        trajs.append(traj)

    # ANALYZE TRAJECTORIES

    n_crossings = []

    # collect early and late crossings

    crossings_early = []
    crossings_late = []

    ts_before = int(T_BEFORE / DT)
    ts_after = int(T_AFTER / DT)

    for traj in trajs:

        starts, onsets, peak_times, offsets, ends = \
            segment_by_threshold(traj['odors'], ANALYSIS_THRESHOLD)[0].T

        n_crossings.append(len(peak_times))

        for ctr, (start, peak_time, end) in enumerate(zip(starts, peak_times, ends)):

            if not (H_MIN_PEAK <= traj['headings'][peak_time] < H_MAX_PEAK):

                continue

            if not (X_MIN_PEAK <= traj['xs'][peak_time, 0] < X_MAX_PEAK):

                continue

            crossing = np.nan * np.zeros((ts_before + ts_after,))

            ts_before_crossing = peak_time - start
            ts_after_crossing = end - peak_time

            if ts_before_crossing >= ts_before:

                crossing[:ts_before] = traj['headings'][peak_time - ts_before:peak_time]

            else:

                crossing[ts_before - ts_before_crossing:ts_before] = \
                    traj['headings'][start:peak_time]

            if ts_after_crossing >= ts_after:

                crossing[ts_before:] = traj['headings'][peak_time:peak_time + ts_after]

            else:

                crossing[ts_before:ts_before + ts_after_crossing] = \
                    traj['headings'][peak_time:end]

            if SUBTRACT_PEAK_HEADING:

                crossing -= crossing[ts_before]

            if ctr < EARLY_LESS_THAN:

                crossings_early.append(crossing)

            else:

                crossings_late.append(crossing)

    n_crossings = np.array(n_crossings)

    crossings_early = np.array(crossings_early)
    crossings_late = np.array(crossings_late)

    t = np.arange(-ts_before, ts_after) * DT

    h_mean_early = np.nanmean(crossings_early, axis=0)
    h_sem_early = nansem(crossings_early, axis=0)

    h_mean_late = np.nanmean(crossings_late, axis=0)
    h_sem_late = nansem(crossings_late, axis=0)

    fig, axs = plt.figure(figsize=(15, 15), tight_layout=True), []

    axs.append(fig.add_subplot(3, 2, 1))
    axs.append(fig.add_subplot(3, 2, 2))

    handles = []

    try:

        handles.append(axs[0].plot(t, h_mean_early, lw=3, color='b', label='early')[0])
        axs[0].fill_between(t, h_mean_early - h_sem_early, h_mean_early + h_sem_early,
            color='b', alpha=0.2)

    except:

        pass

    try:

        handles.append(axs[0].plot(t, h_mean_late, lw=3, color='g', label='late')[0])
        axs[0].fill_between(t, h_mean_late - h_sem_late, h_mean_late + h_sem_late,
            color='g', alpha=0.2)

    except:

        pass

    axs[0].axvline(0, ls='--', color='gray')

    axs[0].set_xlabel('time since peak (s)')

    if SUBTRACT_PEAK_HEADING:

        axs[0].set_ylabel('change in heading (deg)')

    else:

        axs[0].set_ylabel('heading (deg)')

    axs[0].legend(handles=handles, fontsize=16)

    bin_min = -0.5
    bin_max = n_crossings.max() + 0.5

    bins = np.linspace(bin_min, bin_max, bin_max - bin_min + 1, endpoint=True)

    axs[1].hist(n_crossings, bins=bins, lw=0, normed=True)
    axs[1].set_xlim(bin_min, bin_max)

    axs[1].set_xlabel('number of crossings')
    axs[1].set_ylabel('proportion of trajectories')

    axs.append(fig.add_subplot(3, 1, 2))

    axs[2].plot(trajs[0]['xs'][:, 0], trajs[0]['xs'][:, 1])
    axs[2].axhline(0, color='gray', ls='--')

    axs[2].set_xlabel('x (m)')
    axs[2].set_ylabel('y (m)')

    axs.append(fig.add_subplot(3, 1, 3))

    all_xy = np.concatenate([traj['xs'][:, :2] for traj in trajs[:3000]], axis=0)
    x_bins = np.linspace(BOUNDS[0][0], BOUNDS[0][1], 66, endpoint=True)
    y_bins = np.linspace(BOUNDS[1][0], BOUNDS[1][1], 30, endpoint=True)

    axs[3].hist2d(all_xy[:, 0], all_xy[:, 1], bins=(x_bins, y_bins))

    axs[3].set_xlabel('x (m)')
    axs[3].set_ylabel('y (m)')

    for ax in axs:

        set_font_size(ax, 20)

    return fig


def crossing_triggered_headings_early_late_vary_param(
        SEED, SAVE_FILE, N_TRAJS, DURATION, DT,
        TAU, NOISE, BIAS, HIT_INFLUENCE, SQRT_K_0,
        VARIABLE_PARAMS, BOUNDS,
        PL_CONC, PL_MEAN, PL_STD,
        H_MIN_PEAK, H_MAX_PEAK,
        X_MIN_PEAK, X_MAX_PEAK,
        EARLY_LESS_THAN,
        SUBTRACT_PEAK_HEADING, T_BEFORE, T_AFTER,
        T_INT_START, T_INT_END, AX_GRID):
    """
    Fly several agents through a simulated plume and plot their plume-crossing-triggered
    headings.
    """

    # try to open saved results

    if os.path.isfile(SAVE_FILE):

        print('Results file found. Loading results file.')
        results = np.load(SAVE_FILE)

    else:

        print('Results file not found. Running analysis...')
        np.random.seed(SEED)

        # build plume

        pl = GaussianLaminarPlume(PL_CONC, PL_MEAN, PL_STD)

        # loop over all parameter sets

        varying_params = []
        fixed_params = []

        early_late_heading_diffs_all = []
        early_late_heading_diffs_lb_all = []
        early_late_heading_diffs_ub_all = []

        for variable_params in VARIABLE_PARAMS:

            print('Variable params: {}'.format(variable_params))

            assert set(variable_params.keys()) == set(
                ['threshold', 'tau_memory', 'sqrt_k_s'])

            # identify which parameter is varying

            for key, vals in variable_params.items():

                if isinstance(vals, list):

                    varying_params.append((key, vals))
                    fixed_params.append((
                        [(k, v) for k, v in variable_params.items() if k != key]))

                    n_param_sets = len(vals)

                    break

            # make other parameters into lists so they can all be looped over nicely

            for key, vals in variable_params.items():

                if not isinstance(vals, list):

                    variable_params[key] = [vals for _ in range(n_param_sets)]

            early_late_heading_diffs = []
            early_late_heading_diffs_lb = []
            early_late_heading_diffs_ub = []

            for param_set_ctr in range(len(variable_params.values()[0])):

                threshold = variable_params['threshold'][param_set_ctr]
                hit_influence = HIT_INFLUENCE
                tau_memory = variable_params['tau_memory'][param_set_ctr]
                k_0 = np.array([
                    [SQRT_K_0 ** 2, 0],
                    [0, SQRT_K_0 ** 2],
                ])
                k_s = np.array([
                    [variable_params['sqrt_k_s'][param_set_ctr] ** 2, 0],
                    [0, variable_params['sqrt_k_s'][param_set_ctr] ** 2],
                ])

                # build tracking agent

                ag = CenterlineInferringAgent(
                    tau=TAU, noise=NOISE, bias=BIAS, threshold=threshold,
                    hit_trigger='peak', hit_influence=hit_influence, tau_memory=tau_memory,
                    k_0=k_0, k_s=k_s, bounds=BOUNDS)

                trajs = []

                for _ in range(N_TRAJS):

                    # choose random start position

                    start_pos = np.array([
                        np.random.uniform(*BOUNDS[0]),
                        np.random.uniform(*BOUNDS[1]),
                        np.random.uniform(*BOUNDS[2]),
                    ])

                    # make trajectory

                    traj = ag.track(plume=pl, start_pos=start_pos, duration=DURATION, dt=DT)

                    traj['headings'] = heading(traj['vs'])[:, 2]

                    trajs.append(traj)

                crossings_early = []
                crossings_late = []

                ts_before = int(T_BEFORE / DT)
                ts_after = int(T_AFTER / DT)

                for traj in trajs:

                    starts, onsets, peak_times, offsets, ends = \
                        segment_by_threshold(traj['odors'], threshold)[0].T

                    for ctr, (start, peak_time, end) in enumerate(zip(starts, peak_times, ends)):

                        if not (H_MIN_PEAK <= traj['headings'][peak_time] < H_MAX_PEAK):

                            continue

                        if not (X_MIN_PEAK <= traj['xs'][peak_time, 0] < X_MAX_PEAK):

                            continue

                        crossing = np.nan * np.zeros((ts_before + ts_after,))

                        ts_before_crossing = peak_time - start
                        ts_after_crossing = end - peak_time

                        if ts_before_crossing >= ts_before:

                            crossing[:ts_before] = traj['headings'][peak_time - ts_before:peak_time]

                        else:

                            crossing[ts_before - ts_before_crossing:ts_before] = \
                                traj['headings'][start:peak_time]

                        if ts_after_crossing >= ts_after:

                            crossing[ts_before:] = traj['headings'][peak_time:peak_time + ts_after]

                        else:

                            crossing[ts_before:ts_before + ts_after_crossing] = \
                                traj['headings'][peak_time:end]

                        if SUBTRACT_PEAK_HEADING:

                            crossing -= crossing[ts_before]

                        if ctr < EARLY_LESS_THAN:

                            crossings_early.append(crossing)

                        else:

                            crossings_late.append(crossing)

                crossings_early = np.array(crossings_early)
                crossings_late = np.array(crossings_late)

                t = np.arange(-ts_before, ts_after) * DT

                h_mean_early = np.nanmean(crossings_early, axis=0)
                h_mean_late = np.nanmean(crossings_late, axis=0)

                h_sem_early = nansem(crossings_early, axis=0)
                h_sem_late = nansem(crossings_late, axis=0)

                h_mean_diff = h_mean_late - h_mean_early

                h_mean_diff_lb = h_mean_late - h_sem_late - (h_mean_early + h_sem_early)
                h_mean_diff_ub = h_mean_late + h_sem_late - (h_mean_early - h_sem_early)

                early_late_heading_diff = \
                    h_mean_diff[(t > T_INT_START) * (t <= T_INT_END)].mean()
                early_late_heading_diff_lb = \
                    h_mean_diff_lb[(t > T_INT_START) * (t <= T_INT_END)].mean()
                early_late_heading_diff_ub = \
                    h_mean_diff_ub[(t > T_INT_START) * (t <= T_INT_END)].mean()

                early_late_heading_diffs.append(early_late_heading_diff)
                early_late_heading_diffs_lb.append(early_late_heading_diff_lb)
                early_late_heading_diffs_ub.append(early_late_heading_diff_ub)

            early_late_heading_diffs_all.append(np.array(early_late_heading_diffs))
            early_late_heading_diffs_lb_all.append(np.array(early_late_heading_diffs_lb))
            early_late_heading_diffs_ub_all.append(np.array(early_late_heading_diffs_ub))

        # save results

        results = np.array([
            {
                'varying_params': varying_params,
                'fixed_params': fixed_params,
                'early_late_heading_diffs_all': early_late_heading_diffs_all,
                'early_late_heading_diffs_lb_all': early_late_heading_diffs_lb_all,
                'early_late_heading_diffs_ub_all': early_late_heading_diffs_ub_all,
             }])
        np.save(SAVE_FILE, results)

    results = results[0]

    ## MAKE PLOTS

    fig_size = (5 * AX_GRID[1], 4 * AX_GRID[0])
    fig, axs = plt.subplots(*AX_GRID, figsize=fig_size, tight_layout=True)

    for ax_ctr in range(len(results['varying_params'])):

        ax = axs.flatten()[ax_ctr]

        ys_plot = results['early_late_heading_diffs_all'][ax_ctr]
        ys_err = [
            ys_plot - results['early_late_heading_diffs_lb_all'][ax_ctr],
            results['early_late_heading_diffs_ub_all'][ax_ctr] - ys_plot
        ]

        xs_name = results['varying_params'][ax_ctr][0]
        xs_plot = np.arange(len(ys_plot))

        ax.errorbar(
            xs_plot, ys_plot, yerr=ys_err, color='k', fmt='--o')

        ax.axhline(0, color='gray')

        if np.max(results['early_late_heading_diffs_ub_all'][ax_ctr]) > 0:

            y_range = np.max(results['early_late_heading_diffs_ub_all'][ax_ctr]) - \
                      np.min(results['early_late_heading_diffs_lb_all'][ax_ctr])

        else:

            y_range = -np.min(results['early_late_heading_diffs_lb_all'][ax_ctr])

        y_min = np.min(
            results['early_late_heading_diffs_lb_all'][ax_ctr]) - 0.1 * y_range
        y_max = max(np.max(
            results['early_late_heading_diffs_ub_all'][ax_ctr]), 0) + 0.1 * y_range

        ax.set_xlim(-1, len(ys_plot))
        ax.set_xticks(xs_plot)
        
        x_ticklabels = results['varying_params'][ax_ctr][1]
        
        if xs_name == 'threshold': 
            
            x_ticklabels = ['{0:.4f}'.format(xtl * (0.0476/526)) for xtl in x_ticklabels]
            
        ax.set_xticklabels(x_ticklabels)

        ax.set_ylim(y_min, y_max)

        if xs_name == 'tau_memory': x_label = 'tau_m (s)'
        elif xs_name == 'threshold': x_label = 'threshold (% ethanol)'
        else: x_label = xs_name
            
        ax.set_xlabel(x_label)
        ax.set_ylabel('mean heading difference\nfor late vs. early crossings')

    for ax in axs.flatten():

        set_font_size(ax, 16)

    return fig
