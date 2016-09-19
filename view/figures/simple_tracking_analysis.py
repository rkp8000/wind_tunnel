from __future__ import division, print_function
from itertools import product as cproduct
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

    h_mean_early = np.nanmean(crossings_early, axis=0)
    h_sem_early = nansem(crossings_early, axis=0)

    h_mean_late = np.nanmean(crossings_late, axis=0)
    h_sem_late = nansem(crossings_late, axis=0)

    speeds = np.concatenate(
        [np.linalg.norm(traj['vs'], axis=1) for traj in trajs])

    fig, axs = plt.figure(figsize=(15, 10), tight_layout=True), []

    axs.append(fig.add_subplot(2, 2, 1))
    axs.append(fig.add_subplot(2, 2, 2))

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

    axs[1].hist(speeds, bins=50, lw=0, normed=True)

    axs[1].set_xlabel('speed (m/s)')
    axs[1].set_ylabel('proportion of time points')

    axs.append(fig.add_subplot(2, 1, 2))

    axs[2].plot(trajs[0]['xs'][:, 0], trajs[0]['xs'][:, 1])
    axs[2].axhline(0, color='gray', ls='--')

    axs[2].set_xlabel('x')
    axs[2].set_ylabel('y')

    for ax in axs:

        set_font_size(ax, 20)

    return fig


def crossing_triggered_headings_early_late_vs_bias(
        SEED,
        N_TRAJS, DURATION, DT, START_POS_RANGE,
        PL_CONC, PL_MEAN, PL_K,
        TAU, NOISE, BIAS, THRESHOLD,
        HIT_TRIGGER, HIT_INFLUENCE,
        K_0, K_S,
        H_MIN_PEAK, H_MAX_PEAK,
        EARLY_LESS_THAN,
        SUBTRACT_PEAK_HEADING, T_BEFORE, T_AFTER,
        T_INT_START, T_INT_END):
    """
    Fly several agents through a simulated plume and plot their plume-crossing-triggered
    headings.
    """

    if len(BIAS) > 1:

        vary = 'bias'
        x_plot = BIAS

    elif len(HIT_INFLUENCE) > 1:

        vary = 'hit_influence'
        x_plot = HIT_INFLUENCE

    elif len(THRESHOLD) > 1:

        vary = 'threshold'
        x_plot = THRESHOLD

    elif len(K_S) > 1:

        vary = 'k_s'
        x_plot = K_S

    else:

        raise Exception('One of the parameters must vary.')

    param_sets = cproduct(BIAS, HIT_INFLUENCE, THRESHOLD, K_S)

    # generate trajectories

    np.random.seed(SEED)

    early_late_heading_diffs = []

    for param_set in param_sets:

        print(param_set)

        bias, hit_influence, threshold = param_set[:3]

        k_s = np.array([
            [param_set[3], 0.],
            [0, param_set[3]]
            ])

        # build plume and agent

        pl = GaussianLaminarPlume(PL_CONC, PL_MEAN, PL_K)
        ag = CenterlineInferringAgent(
            tau=TAU, noise=NOISE, bias=bias, threshold=threshold,
            hit_trigger=HIT_TRIGGER, hit_influence=hit_influence, k_0=K_0, k_s=k_s)

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
                segment_by_threshold(traj['odors'], threshold)[0].T

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

        h_mean_early = np.nanmean(crossings_early, axis=0)
        h_mean_late = np.nanmean(crossings_late, axis=0)

        h_mean_diff = h_mean_late - h_mean_early

        early_late_heading_diff = h_mean_diff[(t > T_INT_START) * (t <= T_INT_END)].mean()

        early_late_heading_diffs.append(early_late_heading_diff)

    early_late_heading_diffs = np.array(early_late_heading_diffs)

    ## MAKE PLOTS

    fig, ax = plt.subplots(1, 1, figsize=(5, 4), tight_layout=True)

    ax.scatter(x_plot, early_late_heading_diffs, color='k', lw=0)
    ax.axhline(0, color='gray', ls='--')

    if np.max(early_late_heading_diffs) > 0:

        y_range = np.max(early_late_heading_diffs) - np.min(early_late_heading_diffs)

    else:

        y_range = -np.min(early_late_heading_diffs)

    y_min = np.min(early_late_heading_diffs) - 0.1 * y_range
    y_max = max(np.max(early_late_heading_diffs), 0) + 0.1 * y_range

    ax.set_ylim(y_min, y_max)

    ax.set_xlabel(vary)
    ax.set_ylabel('mean heading difference\nfor late vs. early crossings')

    set_font_size(ax, 16)

    return fig
