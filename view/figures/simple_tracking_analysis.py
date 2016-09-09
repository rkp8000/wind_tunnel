from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np

from kinematics import heading
from plot import set_font_size
from simple_tracking import GaussianLaminarPlume, CenterlineInferringAgent
from stats import nansem
from time_series import segment_by_threshold


def crossing_triggered_headings_all(
        SEED,
        N_TRAJS, DURATION, DT, START_POS_RANGE,
        PL_CONC, PL_MEAN, PL_K,
        TAU, NOISE, BIAS, AGENT_THRESHOLD,
        HIT_TRIGGER, HIT_INFLUENCE,
        K_0, K_S,
        ANALYSIS_THRESHOLD, H_MIN_PEAK, H_MAX_PEAK,
        SUBTRACT_PEAK_HEADING, T_BEFORE, T_AFTER):
    """
    Fly several agents through a simulated plume and plot their plume-crossing-triggered
    headings.
    """

    # build plume and agent

    pl = GaussianLaminarPlume(PL_CONC, PL_MEAN, PL_K)
    ag = CenterlineInferringAgent(
        tau=TAU, noise=NOISE, bias=BIAS, threshold=AGENT_THRESHOLD,
        hit_trigger=HIT_TRIGGER, hit_influence=HIT_INFLUENCE, k_0=K_0, k_s=K_S)

    # generate trajectories

    np.random.seed(SEED)

    trajs = []

    for _ in range(N_TRAJS):

        # choose random start position

        start_pos = np.array(
            [
                0,
                np.random.uniform(START_POS_RANGE[0][0], START_POS_RANGE[0][1]),
                np.random.uniform(START_POS_RANGE[1][0], START_POS_RANGE[1][1]),
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

    ax.set_xlabel('time since peak (s)')

    if SUBTRACT_PEAK_HEADING:

        ax.set_ylabel('change in heading (deg)')

    else:

        ax.set_ylabel('heading (deg)')

    set_font_size(ax, 16)

    return fig


def crossing_triggered_headings_early_late(
        SEED,
        N_TRAJS, DURATION, DT, START_POS_RANGE,
        PL_CONC, PL_MEAN, PL_K,
        TAU, NOISE, BIAS, AGENT_THRESHOLD,
        HIT_TRIGGER, HIT_INFLUENCE,
        K_0, K_S,
        ANALYSIS_THRESHOLD, H_MIN_PEAK, H_MAX_PEAK,
        EARLY_LESS_THAN,
        SUBTRACT_PEAK_HEADING, T_BEFORE, T_AFTER):
    """
    Fly several agents through a simulated plume and plot their plume-crossing-triggered
    headings.
    """

    # build plume and agent

    pl = GaussianLaminarPlume(PL_CONC, PL_MEAN, PL_K)
    ag = CenterlineInferringAgent(
        tau=TAU, noise=NOISE, bias=BIAS, threshold=AGENT_THRESHOLD,
        hit_trigger=HIT_TRIGGER, hit_influence=HIT_INFLUENCE, k_0=K_0, k_s=K_S)

    # generate trajectories

    np.random.seed(SEED)

    trajs = []

    for _ in range(N_TRAJS):

        # choose random start position

        start_pos = np.array(
            [
                0,
                np.random.uniform(START_POS_RANGE[0][0], START_POS_RANGE[0][1]),
                np.random.uniform(START_POS_RANGE[1][0], START_POS_RANGE[1][1]),
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
            segment_by_threshold(traj['odors'], ANALYSIS_THRESHOLD)[0].T

        for ctr, (start, peak_time, end) in enumerate(zip(starts, peak_times, ends)):

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

            if ctr < EARLY_LESS_THAN:

                crossings_early.append(crossing)

            else:

                crossings_late.append(crossing)

    crossings_early = np.array(crossings_early)
    crossings_late = np.array(crossings_late)

    t = np.arange(-ts_before, ts_after) * DT

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)

    h_mean_early = np.nanmean(crossings_early, axis=0)
    h_sem_early = nansem(crossings_early, axis=0)

    h_mean_late = np.nanmean(crossings_late, axis=0)
    h_sem_late = nansem(crossings_late, axis=0)

    handles = []

    handles.append(ax.plot(t, h_mean_early, lw=3, color='b', label='early')[0])
    ax.fill_between(t, h_mean_early - h_sem_early, h_mean_early + h_sem_early,
        color='b', alpha=0.2)

    handles.append(ax.plot(t, h_mean_late, lw=3, color='g', label='late')[0])
    ax.fill_between(t, h_mean_late - h_sem_late, h_mean_late + h_sem_late,
        color='g', alpha=0.2)

    ax.axvline(0, ls='--', color='gray')

    ax.set_xlabel('time since peak (s)')

    if SUBTRACT_PEAK_HEADING:

        ax.set_ylabel('change in heading (deg)')

    else:

        ax.set_ylabel('heading (deg)')

    ax.legend(handles=handles, fontsize=16)

    set_font_size(ax, 16)

    return fig