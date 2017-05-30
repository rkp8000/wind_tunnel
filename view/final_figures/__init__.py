from __future__ import division, print_function
from copy import deepcopy
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
from random import choice
from pprint import pprint
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ks_2samp, ttest_rel
from scipy.signal import resample
from sklearn import linear_model
import warnings; warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

from db_api import models
from db_api.connect import session

from axis_tools import set_fontsize
from kinematics import heading as calc_heading
from plot import get_n_colors, set_font_size
import simple_models
from simple_tracking import GaussianLaminarPlume
from simple_tracking import CenterlineInferringAgent, SurgingAgent
import stats
import time_series_classifier as tsc
from time_series import get_ks_p_vals, segment_by_threshold

from experimental_constants import DT, PLUME_PARAMS_DICT


def example_traj_and_crossings(
        SEED, EXPT_ID, TRAJ_NUMBER, TRAJ_START_TP, TRAJ_END_TP,
        CROSSING_GROUP, N_CROSSINGS,
        X_0_MIN, X_0_MAX, H_0_MIN, H_0_MAX, MIN_PEAK_CONC,
        TS_BEFORE_3D, TS_AFTER_3D, TS_BEFORE_HEADING, TS_AFTER_HEADING,
        FIG_SIZE, SCATTER_SIZE, CYL_STDS, CYL_COLOR, CYL_ALPHA,
        EXPT_LABEL, FONT_SIZE):
    """
    Show an example trajectory through a wind tunnel plume with the crossings marked.
    Show many crossings overlaid on the plume in 3D and show the mean peak-triggered heading
    with its SEM as well as many individual examples.
    """

    if isinstance(TRAJ_NUMBER, int):
        trajs = session.query(models.Trajectory).filter_by(
            experiment_id=EXPT_ID, odor_state='on', clean=True).all()
        traj = list(trajs)[TRAJ_NUMBER]
    else:
        traj = session.query(models.Trajectory).filter_by(id=TRAJ_NUMBER).first()

    # get plottable quantities
    x_traj, y_traj, z_traj = traj.positions(session).T[:, TRAJ_START_TP:TRAJ_END_TP]
    c_traj = traj.odors(session)[TRAJ_START_TP:TRAJ_END_TP]

    # get several random crossings
    crossings_all = session.query(models.Crossing).filter_by(
        crossing_group_id=CROSSING_GROUP).filter(
        models.Crossing.max_odor > MIN_PEAK_CONC).all()
    crossings_all = list(crossings_all)

    np.random.seed(SEED)
    plot_idxs = np.random.permutation(len(crossings_all))
    crossing_examples = []
    crossing_ctr = 0
    headings = []

    for idx in plot_idxs:
        crossing = crossings_all[idx]

        # throw away crossings that do not meet trigger criteria
        x_0 = getattr(crossing.feature_set_basic, 'position_x_{}'.format('peak'))

        if not (X_0_MIN <= x_0 <= X_0_MAX):
            continue
        h_0 = getattr(crossing.feature_set_basic, 'heading_xyz_{}'.format('peak'))
        if not (H_0_MIN <= h_0 <= H_0_MAX):
            continue

        # store example crossing if desired
        if crossing_ctr < N_CROSSINGS:
            crossing_dict = {}

            for field in ['position_x', 'position_y', 'position_z', 'odor']:
                crossing_dict[field] = crossing.timepoint_field(
                    session, field, -TS_BEFORE_3D, TS_AFTER_3D, 'peak', 'peak')

            crossing_dict['heading'] = crossing.timepoint_field(
                session, 'heading_xyz', -TS_BEFORE_HEADING, TS_AFTER_HEADING - 1,
                'peak', 'peak', nan_pad=True)

            crossing_examples.append(crossing_dict)

        # store crossing heading
        temp = crossing.timepoint_field(
            session, 'heading_xyz', -TS_BEFORE_HEADING, TS_AFTER_HEADING - 1,
            'peak', 'peak', nan_pad=True)

        headings.append(temp)

        # increment crossing ctr
        crossing_ctr += 1

    headings = np.array(headings)

    ## MAKE PLOTS
    fig, axs = plt.figure(figsize=FIG_SIZE, tight_layout=True), []

    #  plot example trajectory
    axs.append(fig.add_subplot(3, 1, 1, projection='3d'))

    # overlay plume cylinder
    CYL_MEAN_Y = PLUME_PARAMS_DICT[EXPT_ID]['ymean']
    CYL_MEAN_Z = PLUME_PARAMS_DICT[EXPT_ID]['zmean']

    CYL_SCALE_Y = CYL_STDS * PLUME_PARAMS_DICT[EXPT_ID]['ystd']
    CYL_SCALE_Z = CYL_STDS * PLUME_PARAMS_DICT[EXPT_ID]['zstd']

    MAX_CONC = PLUME_PARAMS_DICT[EXPT_ID]['max_conc']

    y = np.linspace(-1, 1, 100, endpoint=True)
    x = np.linspace(-0.3, 1, 5, endpoint=True)
    yy, xx = np.meshgrid(y, x)
    zz = np.sqrt(1 - yy ** 2)

    yy = CYL_SCALE_Y * yy + CYL_MEAN_Y
    zz_top = CYL_SCALE_Z * zz + CYL_MEAN_Z
    zz_bottom = -CYL_SCALE_Z * zz + CYL_MEAN_Z
    rstride = 20
    cstride = 10

    axs[0].plot_surface(
        xx, yy, zz_top, lw=0,
        color=CYL_COLOR, alpha=CYL_ALPHA, rstride=rstride, cstride=cstride)

    axs[0].plot_surface(
        xx, yy, zz_bottom, lw=0,
        color=CYL_COLOR, alpha=CYL_ALPHA, rstride=rstride, cstride=cstride)

    axs[0].plot(x_traj, y_traj, z_traj, color='k', lw=3, zorder=1)
    axs[0].scatter(
        x_traj, y_traj, z_traj, c=c_traj, s=SCATTER_SIZE,
        vmin=0, vmax=MAX_CONC/2, cmap=cmx.hot, lw=0, alpha=1)
    # mark start
    axs[0].scatter(
        x_traj[0], y_traj[0], z_traj[0], s=400, marker='*', lw=0, c='g',
        zorder=3
    )

    axs[0].set_xlim(-0.3, 1)
    axs[0].set_ylim(-0.15, 0.15)
    axs[0].set_zlim(-0.15, 0.15)

    axs[0].set_xticks([-0.3, 1.])
    axs[0].set_yticks([-0.15, 0.15])
    axs[0].set_zticks([-0.15, 0.15])

    axs[0].set_xticklabels([-30, 100])
    axs[0].set_yticklabels([-15, 15])
    axs[0].set_zticklabels([-15, 15])

    axs[0].set_xlabel('x (cm)')
    axs[0].set_ylabel('y (cm)')
    axs[0].set_zlabel('z (cm)')

    # plot several crossings
    axs.append(fig.add_subplot(3, 1, 2, projection='3d'))

    # overlay plume cylinder
    axs[1].plot_surface(
        xx, yy, zz_top, lw=0,
        color=CYL_COLOR, alpha=CYL_ALPHA, rstride=rstride, cstride=cstride)

    axs[1].plot_surface(
        xx, yy, zz_bottom, lw=0,
        color=CYL_COLOR, alpha=CYL_ALPHA, rstride=rstride, cstride=cstride)

    # plot crossings
    for crossing in crossing_examples:

        axs[1].scatter(
            crossing['position_x'], crossing['position_y'], crossing['position_z'],
            c=crossing['odor'], s=SCATTER_SIZE,
            vmin=0, vmax=MAX_CONC / 2, cmap=cmx.hot, lw=0, alpha=1)

    axs[1].set_xlim(-0.3, 1)
    axs[1].set_ylim(-0.15, 0.15)
    axs[1].set_zlim(-0.15, 0.15)

    axs[1].set_xticks([-0.3, 1.])
    axs[1].set_yticks([-0.15, 0.15])
    axs[1].set_zticks([-0.15, 0.15])

    axs[1].set_xticklabels([-30, 100])
    axs[1].set_yticklabels([-15, 15])
    axs[1].set_zticklabels([-15, 15])

    axs[1].set_xlabel('x (cm)')
    axs[1].set_ylabel('y (cm)')
    axs[1].set_zlabel('z (cm)')

    # plot headings
    axs.append(fig.add_subplot(3, 2, 6))

    t = np.arange(-TS_BEFORE_HEADING, TS_AFTER_HEADING) * DT
    headings_mean = np.nanmean(headings, axis=0)
    headings_std = np.nanstd(headings, axis=0)
    headings_sem = stats.nansem(headings, axis=0)

    # plot example crossings
    for crossing in crossing_examples:
        axs[2].plot(t, crossing['heading'], lw=1, color='k', alpha=0.5, zorder=-1)

    # plot mean, sem, and std
    axs[2].plot(t, headings_mean, lw=3, color='k', zorder=1)
    axs[2].plot(t, headings_mean - headings_std, lw=3, ls='--', color='k', zorder=1)
    axs[2].plot(t, headings_mean + headings_std, lw=3, ls='--', color='k', zorder=1)
    axs[2].fill_between(
        t, headings_mean - headings_sem, headings_mean + headings_sem,
        color='k', alpha=0.2)

    axs[2].set_xlabel('time since crossing (s)')
    axs[2].set_ylabel('heading (degrees)')

    for ax in axs:
        set_fontsize(ax, FONT_SIZE)

    return fig


def heading_concentration_dependence(
        CROSSING_GROUP_IDS, CROSSING_GROUP_LABELS,
        X_0_MIN, X_0_MAX, H_0_MIN, H_0_MAX,
        T_BEFORE, T_AFTER,
        T_MODELS,
        CROSSING_GROUP_EXAMPLE_ID,
        FIG_SIZE, CROSSING_GROUP_COLORS,
        SCATTER_SIZE, SCATTER_COLOR, SCATTER_ALPHA,
        FONT_SIZE):
    """
    Show a partial correlation plot between concentration and heading a little while after the
    peak odor concentration. Show the relationship between peak concentration and heading
    at a specific time (T_MODEL) post peak via a scatter plot.

    Then fit a binary threshold model and a model with a linear piece to the data and see if
    the linear one fits significantly better.
    """

    conversion_factor = 0.0476 / 526

    ## CALCULATE PARTIAL CORRELATIONS

    # convert times to timesteps
    ts_before = int(round(T_BEFORE / DT))
    ts_after = int(round(T_AFTER / DT))
    ts_models = {
        cg_id: int(round(t_model / DT))
        for cg_id, t_model in T_MODELS.items()
    }

    data = {cg_id: None for cg_id in CROSSING_GROUP_IDS}

    for cg_id in CROSSING_GROUP_IDS:

        # get crossing group and crossings
        crossings_all = session.query(models.Crossing).join(
            models.CrossingFeatureSetBasic).filter(
            models.Crossing.crossing_group_id == cg_id,
            models.CrossingFeatureSetBasic.position_x_peak.between(
                X_0_MIN, X_0_MAX),
            models.CrossingFeatureSetBasic.heading_xyz_peak.between(
                H_0_MIN, H_0_MAX))

        # get all initial hs, initial xs, peak concs, and heading time-series
        x_0s = []
        h_0s = []
        c_maxs = []
        headings = []
        trajs_unique = []

        for crossing in crossings_all:

            c_maxs.append(crossing.max_odor * conversion_factor)
            x_0s.append(crossing.feature_set_basic.position_x_peak)
            h_0s.append(crossing.feature_set_basic.heading_xyz_peak)

            headings_ = crossing.timepoint_field(
                session, 'heading_xyz', -ts_before, ts_after - 1,
                'peak', 'peak', nan_pad=True)

            headings.append(headings_)

            if crossing.trajectory_id not in trajs_unique:
                trajs_unique.append(crossing.trajectory_id)

        x_0s = np.array(x_0s)
        h_0s = np.array(h_0s)
        c_maxs = np.array(c_maxs)
        headings = np.array(headings)
        n_total = len(trajs_unique)

        partial_corrs = np.nan * np.ones((headings.shape[1],), dtype=float)
        p_vals = np.nan * np.ones((headings.shape[1],), dtype=float)
        lbs = np.nan * np.ones((headings.shape[1],), dtype=float)
        ubs = np.nan * np.ones((headings.shape[1],), dtype=float)
        ns = np.nan * np.ones((headings.shape[1],), dtype=float)

        # loop through all time steps
        for t_step in range(headings.shape[1]):
            headings_this_tp = headings[:, t_step]

            if t_step == (ts_models[cg_id] + ts_before):
                model_headings = headings_this_tp.copy()

            # create not-nan mask
            mask = ~np.isnan(headings_this_tp)
            ns[t_step] = min(mask.sum(), n_total)

            # get partial correlations using all not-nan values
            r, p, lb, ub = stats.pearsonr_partial_with_confidence(
                c_maxs[mask], headings_this_tp[mask],
                [x_0s[mask], h_0s[mask]], n=ns[t_step])

            partial_corrs[t_step] = r
            p_vals[t_step] = p
            lbs[t_step] = lb
            ubs[t_step] = ub

        data[cg_id] = {
            'x_0s': x_0s,
            'h_0s': h_0s,
            'c_maxs': c_maxs,
            'headings': headings,
            'partial_corrs': partial_corrs,
            'p_vals': p_vals,
            'lbs': lbs,
            'ubs': ubs,
            'model_headings': model_headings,
            'n_total': n_total,
        }

    ## MAKE PLOT OF PARTIAL CORRELATIONS
    fig, axs = plt.figure(figsize=FIG_SIZE, facecolor='white', tight_layout=True), []

    axs.append(fig.add_subplot(2, 1, 1))

    t = np.arange(-ts_before, ts_after) * DT
    t[ts_before] = np.nan

    handles = []

    for cg_id in CROSSING_GROUP_IDS:
        color = CROSSING_GROUP_COLORS[cg_id]
        label = CROSSING_GROUP_LABELS[cg_id]

        # show partial correlation and confidence
        handle = axs[0].plot(
            t, data[cg_id]['partial_corrs'], color=color,
            lw=2, ls='-', label=label)[0]
        axs[0].fill_between(
            t, data[cg_id]['lbs'], data[cg_id]['ubs'], color=color, alpha=0.2)

        handles.append(handle)

    axs[0].axhline(0, color='gray', ls='--')
    axs[0].set_xlim(-T_BEFORE, T_AFTER)

    axs[0].set_xlabel('time of heading measurement\nsince crossing (s)')
    axs[0].set_ylabel('heading-concentration\npartial correlation')
    axs[0].legend(handles=handles, loc='upper left')

    ## FIT BOTH MODELS TO EACH DATASET
    model_infos = {cg_id: None for cg_id in CROSSING_GROUP_IDS}

    for cg_id in CROSSING_GROUP_IDS:

        hs = data[cg_id]['model_headings']
        c_maxs = data[cg_id]['c_maxs']
        x_0s = data[cg_id]['x_0s']
        h_0s = data[cg_id]['h_0s']

        valid_mask = ~np.isnan(hs)

        hs = hs[valid_mask]
        c_maxs = c_maxs[valid_mask]
        x_0s = x_0s[valid_mask]
        h_0s = h_0s[valid_mask]

        n = min(len(hs), data[cg_id]['n_total'])
        rho = stats.pearsonr_partial_with_confidence(c_maxs, hs, [x_0s, h_0s])[0]
        binary_model = simple_models.ThresholdLinearHeadingConcModel(
            include_c_max_coefficient=False)

        binary_model.brute_force_fit(hs=hs, c_maxs=c_maxs, x_0s=x_0s, h_0s=h_0s)
        hs_predicted_binary = binary_model.predict(c_maxs=c_maxs, x_0s=x_0s, h_0s=h_0s)
        rss_binary = np.sum((hs - hs_predicted_binary) ** 2)

        threshold_linear_model = simple_models.ThresholdLinearHeadingConcModel(
            include_c_max_coefficient=True)
        threshold_linear_model.brute_force_fit(
            hs=hs, c_maxs=c_maxs, x_0s=x_0s, h_0s=h_0s)

        hs_predicted_threshold_linear = threshold_linear_model.predict(
            c_maxs=c_maxs, x_0s=x_0s, h_0s=h_0s)
        rss_threshold_linear = np.sum((hs - hs_predicted_threshold_linear) ** 2)

        f, p_val = stats.f_test(
            rss_reduced=rss_binary, rss_full=rss_threshold_linear,
            df_reduced=7, df_full=8, n=n
        )

        model_infos[cg_id] = {
            'n': n,
            'rss_binary': rss_binary,
            'rss_binary_linear': rss_threshold_linear,
            'f': f,
            'p_val': p_val,
            'threshold_binary': binary_model.threshold,
            'threshold_binary_linear': threshold_linear_model.threshold,
            'h_vs_c_coef': threshold_linear_model.linear_models['above'].coef_[0],
            'rho': rho,
        }

        pprint('Model fit analysis for "{}":'.format(cg_id))
        pprint(model_infos[cg_id])

    axs.append(fig.add_subplot(2, 1, 2))

    axs[-1].scatter(
        data[CROSSING_GROUP_EXAMPLE_ID]['c_maxs'],
        data[CROSSING_GROUP_EXAMPLE_ID]['model_headings'],
        s=SCATTER_SIZE, c=SCATTER_COLOR, lw=0, alpha=SCATTER_ALPHA)

    axs[-1].set_xlim(0, data[CROSSING_GROUP_EXAMPLE_ID]['c_maxs'].max())
    axs[-1].set_ylim(0, 180)

    axs[-1].set_xlabel('concentration (% ethanol)')
    axs[-1].set_ylabel('heading at {} s\n since crossing'.format(
        T_MODELS[CROSSING_GROUP_EXAMPLE_ID]))
    axs[-1].set_title('heading-concentration relationship for {}'.format(
        CROSSING_GROUP_LABELS[CROSSING_GROUP_EXAMPLE_ID]))

    for ax in axs:
        set_fontsize(ax, FONT_SIZE)

    return fig


def early_vs_late_heading_timecourse(
        CROSSING_GROUP_IDS, CROSSING_GROUP_LABELS,
        X_0_MIN, X_0_MAX, H_0_MIN, H_0_MAX,
        MAX_CROSSINGS_EARLY, SUBTRACT_INITIAL_HEADING,
        T_BEFORE, T_AFTER, T_AVG_DIFF_START, T_AVG_DIFF_END,
        AX_SIZE, AX_GRID, EARLY_LATE_COLORS, ALPHA, Y_LIM,
        P_VAL_COLOR, P_VAL_Y_LIM, LEGEND_CROSSING_GROUP_ID,
        X_0_BINS, FONT_SIZE, SAVE_FILE_PREFIX):
    """
    Show early vs. late headings for different experiments, along with a plot of the
    p-values for the difference between the two means.
    """

    # convert times to time steps
    ts_before = int(round(T_BEFORE / DT))
    ts_after = int(round(T_AFTER / DT))

    # loop over crossing groups
    x_0s_dict = {}
    headings_dict = {}
    p_vals_dict = {}

    for cg_id in CROSSING_GROUP_IDS:

        # get early and late crossings
        crossings_dict = {}
        crossings_all = session.query(models.Crossing).join(
            models.CrossingFeatureSetBasic).filter(
            models.Crossing.crossing_group_id == cg_id,
            models.CrossingFeatureSetBasic.position_x_peak.between(
                X_0_MIN, X_0_MAX),
            models.CrossingFeatureSetBasic.heading_xyz_peak.between(
                H_0_MIN, H_0_MAX))
        crossings_dict['early'] = crossings_all.filter(
            models.Crossing.crossing_number <= MAX_CROSSINGS_EARLY)
        crossings_dict['late'] = crossings_all.filter(
            models.Crossing.crossing_number > MAX_CROSSINGS_EARLY)

        x_0s_dict[cg_id] = {}
        headings_dict[cg_id] = {}

        for label in ['early', 'late']:

            x_0s = []
            headings = []

            # get all initial hs, initial xs, peak concs, and heading time-series
            for crossing in crossings_dict[label]:

                # throw away crossings that do not meet trigger criteria
                x_0 = getattr(
                    crossing.feature_set_basic, 'position_x_{}'.format('peak'))

                # store x_0 (uw/dw position)
                x_0s.append(x_0)

                # get and store headings
                temp = crossing.timepoint_field(
                    session, 'heading_xyz', -ts_before, ts_after - 1,
                    'peak', 'peak', nan_pad=True)

                # subtract initial heading if desired
                if SUBTRACT_INITIAL_HEADING:
                    temp -= temp[ts_before]

                # store headings
                headings.append(temp)

            x_0s_dict[cg_id][label] = np.array(x_0s)
            headings_dict[cg_id][label] = np.array(headings)

        # loop through all time points and calculate KS p-val between early and late
        p_vals_dict[cg_id] = get_ks_p_vals(
            headings_dict[cg_id]['early'], headings_dict[cg_id]['late'])

    ## MAKE PLOTS
    t = np.arange(-ts_before, ts_after) * DT

    # history-dependence
    fig_size = (AX_SIZE[0] * AX_GRID[1], AX_SIZE[1] * AX_GRID[0])
    fig, axs = plt.subplots(*AX_GRID, figsize=fig_size, tight_layout=True)

    for cg_id, ax in zip(CROSSING_GROUP_IDS, axs.flat):

        # get mean and sem of headings for early and late groups
        handles = []
        save_data = {'t': t}

        for label, color in EARLY_LATE_COLORS.items():

            headings_mean = np.nanmean(headings_dict[cg_id][label], axis=0)
            headings_sem = stats.nansem(headings_dict[cg_id][label], axis=0)

            handles.append(ax.plot(
                t, headings_mean, color=color, lw=2, label=label, zorder=1)[0])
            ax.fill_between(
                t, headings_mean - headings_sem, headings_mean + headings_sem,
                color=color, alpha=ALPHA, zorder=1)

            save_data[label] = headings_mean.copy()

        save_file = '{}_{}.npy'.format(SAVE_FILE_PREFIX, cg_id)
        np.save(save_file, np.array([save_data]))

        ax.set_ylim(Y_LIM)
        ax.set_xlabel('time since crossing (s)')

        if SUBTRACT_INITIAL_HEADING:
            ax.set_ylabel('$\Delta$ heading (deg.)')
        else:
            ax.set_ylabel('heading (deg.)')

        ax.set_title(CROSSING_GROUP_LABELS[cg_id])

        if cg_id == LEGEND_CROSSING_GROUP_ID:
            ax.legend(handles=handles, loc='upper right')

        set_fontsize(ax, FONT_SIZE)

    return fig


def early_late_heading_timecourse_surge_cast(
        SEED, N_TRAJS, DURATION, DT, BOUNDS,
        TAU, NOISE, BIAS, AGENT_THRESHOLD,
        SURGE_AMP, TAU_SURGE,
        PL_CONC, PL_MEAN, PL_STD,
        ANALYSIS_THRESHOLD,
        H_0_MIN, H_0_MAX,
        X_0_MIN, X_0_MAX,
        MAX_CROSSINGS_EARLY,
        SUBTRACT_PEAK_HEADING, T_BEFORE, T_AFTER,
        Y_LIM, SAVE_FILE):
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
        traj['headings'] = calc_heading(traj['vs'])[:, 2]
        trajs.append(traj)

    # ANALYZE TRAJECTORIES
    n_crossings = []

    # collect early and late crossings
    crossings_early = []
    crossings_late = []

    crossings_save = []

    ts_before = int(T_BEFORE / DT)
    ts_after = int(T_AFTER / DT)

    for traj in trajs:

        starts, onsets, peak_times, offsets, ends = \
            segment_by_threshold(traj['odors'], ANALYSIS_THRESHOLD)[0].T

        n_crossings.append(len(peak_times))

        for ctr, (start, peak_time, end) in enumerate(zip(starts, peak_times, ends)):

            # skip crossings that don't meet inclusion criteria
            if not (H_0_MIN <= traj['headings'][peak_time] <= H_0_MAX):
                continue
            if not (X_0_MIN <= traj['xs'][peak_time, 0] <= X_0_MAX):
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
            if ctr + 1 <= MAX_CROSSINGS_EARLY:
                crossings_early.append(crossing)
            else:
                crossings_late.append(crossing)

            crossings_save.append((ctr + 1, crossing.copy()))

    # save crossings
    save_dict_full = {
        'ts_before': ts_before,
        'ts_after': ts_after,
        'crossings': crossings_save
    }
    save_file = SAVE_FILE + '_full.npy'
    np.save(save_file, np.array([save_dict_full]))

    n_crossings = np.array(n_crossings)

    crossings_early = np.array(crossings_early)
    crossings_late = np.array(crossings_late)

    t = np.arange(-ts_before, ts_after) * DT

    h_mean_early = np.nanmean(crossings_early, axis=0)
    h_sem_early = stats.nansem(crossings_early, axis=0)

    h_mean_late = np.nanmean(crossings_late, axis=0)
    h_sem_late = stats.nansem(crossings_late, axis=0)

    save_data = {'t': t, 'early': h_mean_early, 'late': h_mean_late}
    np.save(SAVE_FILE + '.npy', np.array([save_data]))

    fig, axs = plt.figure(figsize=(14, 15), tight_layout=True), []

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

    axs[0].set_ylim(Y_LIM)
    axs[0].set_xlabel('time since crossing (s)')
    axs[0].set_title('surge-cast')

    if SUBTRACT_PEAK_HEADING:
        axs[0].set_ylabel('$\Delta$ heading (deg)')
    else:
        axs[0].set_ylabel('heading (deg)')

    axs[0].legend(handles=handles, fontsize=16)

    bin_min = -0.5
    bin_max = n_crossings.max() + 0.5

    bins = np.linspace(bin_min, bin_max, bin_max - bin_min + 1, endpoint=True)

    axs[1].hist(n_crossings, bins=bins, lw=0, normed=True)
    axs[1].set_xlim(bin_min, bin_max)

    axs[1].set_xlabel('number of crossings')
    axs[1].set_ylabel('proportion of\ntrajectories')

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


def early_late_heading_timecourse_centerline_inferring(
        SEED, N_TRAJS, DURATION, DT,
        TAU, NOISE, BIAS, AGENT_THRESHOLD,
        HIT_INFLUENCE, TAU_MEMORY,
        K_0, K_S, BOUNDS,
        PL_CONC, PL_MEAN, PL_STD,
        ANALYSIS_THRESHOLD,
        H_0_MIN, H_0_MAX,
        X_0_MIN, X_0_MAX,
        MAX_CROSSINGS_EARLY,
        SUBTRACT_PEAK_HEADING, T_BEFORE, T_AFTER, Y_LIM, SAVE_FILE):
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
        traj['headings'] = calc_heading(traj['vs'])[:, 2]
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

            if not (H_0_MIN <= traj['headings'][peak_time] < H_0_MAX):
                continue

            if not (X_0_MIN <= traj['xs'][peak_time, 0] < X_0_MAX):
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
            if ctr + 1 <= MAX_CROSSINGS_EARLY:
                crossings_early.append(crossing)
            else:
                crossings_late.append(crossing)

    n_crossings = np.array(n_crossings)

    crossings_early = np.array(crossings_early)
    crossings_late = np.array(crossings_late)

    t = np.arange(-ts_before, ts_after) * DT

    h_mean_early = np.nanmean(crossings_early, axis=0)
    h_sem_early = stats.nansem(crossings_early, axis=0)

    h_mean_late = np.nanmean(crossings_late, axis=0)
    h_sem_late = stats.nansem(crossings_late, axis=0)

    save_data = {'t': t, 'early': h_mean_early, 'late': h_mean_late}
    np.save(SAVE_FILE, np.array([save_data]))

    fig, axs = plt.figure(figsize=(14, 15), tight_layout=True), []

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

    axs[0].set_ylim(Y_LIM)
    axs[0].set_xlabel('time since crossing (s)')
    axs[0].set_title('centerline-inferring')

    if SUBTRACT_PEAK_HEADING:
        axs[0].set_ylabel('$\Delta$ heading (deg)')
    else:
        axs[0].set_ylabel('heading (deg)')

    bin_min = -0.5
    bin_max = n_crossings.max() + 0.5

    bins = np.linspace(bin_min, bin_max, bin_max - bin_min + 1, endpoint=True)

    axs[1].hist(n_crossings, bins=bins, lw=0, normed=True)
    axs[1].set_xlim(bin_min, bin_max)

    axs[1].set_xlabel('number of crossings')
    axs[1].set_ylabel('proportion of\ntrajectories')

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


def early_vs_late_heading_timecourse_given_x0_and_t_flight(
        CROSSING_GROUP_IDS, CROSSING_GROUP_LABELS,
        X_0_MIN, X_0_MAX, H_0_MIN, H_0_MAX, CROSSING_NUMBER_MAX,
        MAX_CROSSINGS_EARLY, SUBTRACT_INITIAL_HEADING,
        T_BEFORE, T_AFTER, ADJUST_NS,
        AX_SIZE, AX_GRID, EARLY_LATE_COLORS, ALPHA,
        P_VAL_COLOR, P_VAL_Y_LIM, LEGEND_CROSSING_GROUP_ID,
        FONT_SIZE):
    """
    Show early vs. late headings for different experiments, along with a plot of the
    p-values for the difference between the two means.
    """

    # convert times to time steps
    ts_before = int(round(T_BEFORE / DT))
    ts_after = int(round(T_AFTER / DT))

    # loop over crossing groups
    x_0s_dict = {}
    t_flights_dict = {}
    headings_dict = {}
    residuals_dict = {}
    p_vals_dict = {}

    n_trajs_dict = {}

    scatter_ys_dict = {}
    crossing_ns_dict = {}

    for cg_id in CROSSING_GROUP_IDS:

        # get early and late crossings
        crossings_dict = {}
        crossings_all = session.query(models.Crossing).join(
            models.CrossingFeatureSetBasic).filter(
            models.Crossing.crossing_group_id == cg_id,
            models.CrossingFeatureSetBasic.position_x_peak.between(
                X_0_MIN, X_0_MAX),
            models.CrossingFeatureSetBasic.heading_xyz_peak.between(
                H_0_MIN, H_0_MAX))

        crossings_dict['early'] = crossings_all.filter(
            models.Crossing.crossing_number <= MAX_CROSSINGS_EARLY)
        crossings_dict['late'] = crossings_all.filter(
            models.Crossing.crossing_number > MAX_CROSSINGS_EARLY,
            models.Crossing.crossing_number <= CROSSING_NUMBER_MAX)

        x_0s_dict[cg_id] = {}
        t_flights_dict[cg_id] = {}
        headings_dict[cg_id] = {}
        n_trajs_dict[cg_id] = {}

        scatter_ys_dict[cg_id] = {}
        crossing_ns_dict[cg_id] = {}

        for label in ['early', 'late']:

            x_0s = []
            t_flights = []
            headings = []
            scatter_ys = []
            crossing_ns = []

            # get all initial headings, initial xs, peak concentrations,
            # and heading time-series
            unique_trajs = []
            for crossing in crossings_dict[label]:

                assert crossing.crossing_number > 0
                if label == 'early':
                    assert 0 < crossing.crossing_number <= MAX_CROSSINGS_EARLY
                elif label == 'late':
                    assert MAX_CROSSINGS_EARLY < crossing.crossing_number
                    assert crossing.crossing_number <= CROSSING_NUMBER_MAX

                # throw away crossings that do not meet trigger criteria
                x_0 = getattr(
                    crossing.feature_set_basic, 'position_x_{}'.format('peak'))

                # store x_0 (uw/dw position)
                x_0s.append(x_0)

                # store t_flight
                t_flights.append(crossing.t_flight_peak)

                # save which trajectory this came from
                if crossing.trajectory_id not in unique_trajs:
                    unique_trajs.append(crossing.trajectory_id)

                # get and store headings
                temp = crossing.timepoint_field(
                    session, 'heading_xyz', -ts_before, ts_after - 1,
                    'peak', 'peak', nan_pad=True)

                # subtract initial heading if desired
                if SUBTRACT_INITIAL_HEADING: temp -= temp[ts_before]

                # store headings
                headings.append(temp)

                crossing_ns.append(crossing.crossing_number)

            x_0s_dict[cg_id][label] = np.array(x_0s).copy()
            t_flights_dict[cg_id][label] = np.array(t_flights).copy()
            headings_dict[cg_id][label] = np.array(headings).copy()

            n_trajs_dict[cg_id][label] = len(unique_trajs)

            print('CROSSING GROUP "{}"'.format(CROSSING_GROUP_LABELS[cg_id]))
            print('CROSSING SUBSET "{}"'.format(label))
            print('{} CROSSINGS; {} UNIQUE TRAJECTORIES'.format(
                len(x_0s), len(unique_trajs)))

            scatter_ys_dict[cg_id][label] = np.array(scatter_ys).copy()
            crossing_ns_dict[cg_id][label] = np.array(crossing_ns).copy()

        x_early = x_0s_dict[cg_id]['early']
        x_late = x_0s_dict[cg_id]['late']
        t_flight_early = t_flights_dict[cg_id]['early']
        t_flight_late = t_flights_dict[cg_id]['late']
        h_early = headings_dict[cg_id]['early']
        h_late = headings_dict[cg_id]['late']

        x0s_all = np.concatenate([x_early, x_late])
        t_flights_all = np.concatenate([t_flight_early, t_flight_late])
        hs_all = np.concatenate([h_early, h_late], axis=0)

        residuals_dict[cg_id] = {
            'early': np.nan * np.zeros(h_early.shape),
            'late': np.nan * np.zeros(h_late.shape),
        }

        # fit heading linear prediction from x0 at each time point
        # and subtract from original heading
        coefs_all = []
        for t_step in range(ts_before + ts_after):

            # get all headings for this time point
            hs_t = hs_all[:, t_step]
            residuals = np.nan * np.zeros(hs_t.shape)

            # only use headings that exist
            not_nan = ~np.isnan(hs_t)

            # fit linear model
            lm = linear_model.LinearRegression()
            predictors = np.array([x0s_all, t_flights_all]).T
            lm.fit(predictors[not_nan], hs_t[not_nan])

            residuals[not_nan] = hs_t[not_nan] - lm.predict(predictors[not_nan])

            coefs_all.append(lm.coef_.flatten())

            assert np.all(np.isnan(residuals) == np.isnan(hs_t))

            r_early, r_late = np.split(residuals, [len(x_early)])
            residuals_dict[cg_id]['early'][:, t_step] = r_early
            residuals_dict[cg_id]['late'][:, t_step] = r_late

        coefs_all = np.array(coefs_all)
        print('COEFS (X_0, T_FLIGHT) FOR CROSSING GROUP {}'.format(
            CROSSING_GROUP_LABELS[cg_id]))
        print('MEANS: ({0:.5f}, {1:.5f})'.format(*coefs_all.mean(axis=0)))
        print('STDS: ({0:.5f}, {1:.5f})'.format(*coefs_all.std(axis=0)))

        # loop through all time points and calculate p-value (ks-test)
        # between early and late
        p_vals = []

        for t_step in range(ts_before + ts_after):

            early_with_nans = residuals_dict[cg_id]['early'][:, t_step]
            late_with_nans = residuals_dict[cg_id]['late'][:, t_step]

            early_no_nans = early_with_nans[~np.isnan(early_with_nans)]
            late_no_nans = late_with_nans[~np.isnan(late_with_nans)]

            # calculate statistical significance
            if ADJUST_NS:
                n1 = min(n_trajs_dict[cg_id]['early'], len(early_no_nans))
                n2 = min(n_trajs_dict[cg_id]['late'], len(late_no_nans))
            else:
                n1 = len(early_no_nans)
                n2 = len(late_no_nans)
            p_vals.append(stats.ttest_adjusted_ns(
                early_no_nans, late_no_nans, n1, n2, equal_var=False)[1])

        p_vals_dict[cg_id] = p_vals


    ## MAKE PLOTS
    t = np.arange(-ts_before, ts_after) * DT

    # history-dependence
    fig_size = (AX_SIZE[0] * AX_GRID[1], AX_SIZE[1] * AX_GRID[0])
    fig, axs = plt.subplots(*AX_GRID, figsize=fig_size, tight_layout=True)

    for cg_id, ax in zip(CROSSING_GROUP_IDS, axs.flat):

        print('MAKING PLOT FOR CG "{}"'.format(cg_id))

        # get mean and sem of headings for early and late groups
        handles = []

        for label, color in EARLY_LATE_COLORS.items():
            headings_mean = np.nanmean(residuals_dict[cg_id][label], axis=0)
            headings_sem = stats.nansem(residuals_dict[cg_id][label], axis=0)

            print('{} CROSSINGS FOR "{}" GROUP'.format(
                len(residuals_dict[cg_id][label]), label))
            print('MEAN HEADING = {}'.format(headings_mean.mean()))
            print('SEM HEADING = {}'.format(headings_sem.mean()))

            handles.append(ax.plot(
                t, headings_mean, color=color, lw=2, label=label, zorder=1)[0])
            ax.fill_between(
                t, headings_mean - headings_sem, headings_mean + headings_sem,
                color=color, alpha=ALPHA, zorder=1)

        ax.set_ylim(-10, 10)
        ax.set_xlabel('time since crossing (s)')

        if SUBTRACT_INITIAL_HEADING: ax.set_ylabel('heading* (deg.)')
        else: ax.set_ylabel('heading (deg.)')
        ax.set_title(CROSSING_GROUP_LABELS[cg_id])

        if cg_id == LEGEND_CROSSING_GROUP_ID:
            ax.legend(handles=handles, loc='upper right')
        set_fontsize(ax, FONT_SIZE)

        # plot p-value
        ax_twin = ax.twinx()

        ax_twin.plot(t, p_vals_dict[cg_id], color=P_VAL_COLOR, lw=2, ls='--', zorder=0)
        ax_twin.axhline(0.05, ls='-', lw=2, color='gray')

        ax_twin.set_ylim(*P_VAL_Y_LIM)
        ax_twin.set_ylabel('p-value (t-test)', fontsize=FONT_SIZE)

        set_fontsize(ax_twin, FONT_SIZE)

    return fig


def infotaxis_history_dependence(
        WIND_TUNNEL_CG_IDS, INFOTAXIS_WIND_SPEED_CG_IDS, MAX_CROSSINGS,
        INFOTAXIS_HISTORY_DEPENDENCE_CG_IDS,
        MAX_CROSSINGS_EARLY, X_0_MIN, X_0_MAX, H_0_MIN, H_0_MAX,
        X_0_MIN_SIM, X_0_MAX_SIM, X_0_MIN_SIM_HISTORY, X_0_MAX_SIM_HISTORY,
        T_BEFORE_EXPT, T_AFTER_EXPT, TS_BEFORE_SIM, TS_AFTER_SIM,
        HEADING_SMOOTHING_SIM, Y_LIM,
        AX_GRID, AX_SIZE, FONT_SIZE, EXPT_LABELS, EXPT_COLORS, SIM_LABELS,
        SAVE_FILE_PREFIX):
    """
    Show infotaxis-generated trajectories alongside empirical trajectories.
    """

    from db_api.infotaxis import models as models_infotaxis
    from db_api.infotaxis.connect import session as session_infotaxis

    # get headings from infotaxis plume crossings
    headings = {}
    headings['it_hist_dependence'] = {}

    for cg_id in INFOTAXIS_HISTORY_DEPENDENCE_CG_IDS:

        crossings_all = list(session_infotaxis.query(
            models_infotaxis.Crossing).filter_by(
            crossing_group_id=cg_id).all())

        headings['it_hist_dependence'][cg_id] = {'early': [], 'late': []}
        cr_ctr = 0

        for crossing in crossings_all:

            if cr_ctr >= MAX_CROSSINGS:
                break

            # skip this crossing if it doesn't meet our inclusion criteria
            x_0 = crossing.feature_set_basic.position_x_peak
            h_0 = crossing.feature_set_basic.heading_xyz_peak

            if not (X_0_MIN_SIM_HISTORY <= x_0 <= X_0_MAX_SIM_HISTORY):
                continue
            if not (H_0_MIN <= h_0 <= H_0_MAX):
                continue

            # store crossing heading
            temp = crossing.timepoint_field(
                session_infotaxis, 'hxyz', -TS_BEFORE_SIM, TS_AFTER_SIM - 1,
                'peak', 'peak', nan_pad=True)

            temp[~np.isnan(temp)] = gaussian_filter1d(
                temp[~np.isnan(temp)], HEADING_SMOOTHING_SIM)

            # subtract initial heading
            temp -= temp[TS_BEFORE_SIM]

            # store according to its crossing number
            if crossing.crossing_number <= MAX_CROSSINGS_EARLY:
                headings['it_hist_dependence'][cg_id]['early'].append(temp)
            elif crossing.crossing_number > MAX_CROSSINGS_EARLY:
                headings['it_hist_dependence'][cg_id]['late'].append(temp)
            else:
                raise Exception(
                    'crossing number is not early or late for crossing {}'.format(
                    crossing.id))
            cr_ctr += 1

    headings['it_hist_dependence'][cg_id]['early'] = np.array(
        headings['it_hist_dependence'][cg_id]['early'])

    headings['it_hist_dependence'][cg_id]['late'] = np.array(
        headings['it_hist_dependence'][cg_id]['late'])

    ## MAKE PLOTS
    t = np.arange(-TS_BEFORE_SIM, TS_AFTER_SIM)
    fig_size = (AX_GRID[1] * AX_SIZE[1], AX_GRID[0] * AX_SIZE[0])
    fig, axs = plt.subplots(*AX_GRID, figsize=fig_size, tight_layout=True)

    for ax, cg_id in zip(axs.flatten(), INFOTAXIS_HISTORY_DEPENDENCE_CG_IDS):


        mean_early = np.nanmean(headings['it_hist_dependence'][cg_id]['early'], axis=0)
        sem_early = stats.nansem(
            headings['it_hist_dependence'][cg_id]['early'], axis=0)

        mean_late = np.nanmean(headings['it_hist_dependence'][cg_id]['late'], axis=0)
        sem_late = stats.nansem(headings['it_hist_dependence'][cg_id]['late'], axis=0)

        # plot means and sems
        handle_early = ax.plot(
            t, mean_early, lw=3, color='b', zorder=0, label='early')[0]
        ax.fill_between(
            t, mean_early - sem_early, mean_early + sem_early,
            color='b', alpha=0.2)

        handle_late = ax.plot(
            t, mean_late, lw=3, color='g', zorder=0, label='late')[0]
        ax.fill_between(
            t, mean_late - sem_late, mean_late + sem_late,
            color='g', alpha=0.2)

        ax.set_xlabel('time steps since crossing (s)')
        ax.set_title('infotaxis - {}'.format(SIM_LABELS[cg_id]))

        save_data = {'t': t, 'early': mean_early, 'late': mean_late}
        save_file = '{}_{}.npy'.format(SAVE_FILE_PREFIX, cg_id)
        np.save(save_file, np.array([save_data]))

        ax.set_ylim(Y_LIM)

        ax.set_ylabel('$\Delta$ heading (deg)')

    for ax in axs.flatten():
        set_fontsize(ax, FONT_SIZE)

    return fig


def example_trajs_real_and_models(
        EXPT_ID, TRAJ_NUMBER, TRAJ_END_TP, INFOTAXIS_SIMULATION_ID,
        DT, TAU, NOISE, BIAS, THRESHOLD, PL_CONC, PL_MEAN, PL_STD, BOUNDS,
        HIT_INFLUENCE, TAU_MEMORY, K_0, K_S, SURGE_AMP, TAU_SURGE,
        SEED_SURGE, SEED_CENTERLINE,
        FIG_SIZE, SCATTER_SIZE, CYL_STDS, CYL_COLOR, CYL_ALPHA,
        EXPT_LABEL, FONT_SIZE):
    """
    Show an example trajectory through a wind tunnel plume with the crossings marked.
    Show many crossings overlaid on the plume in 3D and show the mean peak-triggered heading
    with its SEM as well as many individual examples.
    """

    from db_api.infotaxis import models as models_infotaxis
    from db_api.infotaxis.connect import session as session_infotaxis

    if isinstance(TRAJ_NUMBER, int):
        trajs = session.query(models.Trajectory).filter_by(
            experiment_id=EXPT_ID, odor_state='on', clean=True).all()
        traj = list(trajs)[TRAJ_NUMBER]
    else:
        traj = session.query(models.Trajectory).filter_by(id=TRAJ_NUMBER).first()

    print('Trajectory: {}'.format(traj.id))

    # get plottable quantities for real trajectory
    xs, ys, zs = traj.positions(session).T[:, :TRAJ_END_TP]
    cs = traj.odors(session)[:TRAJ_END_TP]

    traj_dict_real = {
        'title': 'empirical', 'xs': xs, 'ys': ys, 'zs': zs, 'cs': cs}

    # generate trajectories for simple trackers
    pl = GaussianLaminarPlume(PL_CONC, PL_MEAN, PL_STD)

    start_pos = np.array([xs[0], ys[0], zs[0]])
    duration = TRAJ_END_TP * DT

    # generate surge-cast trajectory
    np.random.seed(SEED_SURGE)
    ag_surge = SurgingAgent(
        tau=TAU, noise=NOISE, bias=BIAS, threshold=THRESHOLD,
        hit_trigger='peak', surge_amp=SURGE_AMP, tau_surge=TAU_SURGE,
        bounds=BOUNDS)
    traj_surge = ag_surge.track(pl, start_pos, duration, DT)
    traj_dict_surge = {
        'title': 'surge-cast',
        'xs': traj_surge['xs'][:, 0],
        'ys': traj_surge['xs'][:, 1],
        'zs': traj_surge['xs'][:, 2],
        'cs': traj_surge['odors'],
    }

    # generate centerline-inferring trajectory
    np.random.seed(SEED_CENTERLINE)
    k_0 = K_0 * np.eye(2)
    k_s = K_S * np.eye(2)

    ag_centerline = CenterlineInferringAgent(
        tau=TAU, noise=NOISE, bias=BIAS, threshold=THRESHOLD,
        hit_trigger='peak', hit_influence=HIT_INFLUENCE,
        k_0=k_0, k_s=k_s, tau_memory=TAU_MEMORY, bounds=BOUNDS)
    traj_centerline = ag_centerline.track(pl, start_pos, duration, DT)
    traj_dict_centerline = {
        'title': 'centerline-inferring',
        'xs': traj_centerline['xs'][:, 0],
        'ys': traj_centerline['xs'][:, 1],
        'zs': traj_centerline['xs'][:, 2],
        'cs': traj_centerline['odors'],
    }

    # get infotaxis trajectory corresponding to real trajectory
    real_trajectory_id = traj.id
    # get geometric configuration corresponding to this real trajectory
    gcert = session_infotaxis.query(
        models_infotaxis.GeomConfigExtensionRealTrajectory).filter_by(
        real_trajectory_id=real_trajectory_id).first()
    gc = gcert.geom_config
    end_tp_info = int(TRAJ_END_TP * DT / gcert.avg_dt)

    trial = session_infotaxis.query(models_infotaxis.Trial).filter(
        models_infotaxis.Trial.simulation_id == INFOTAXIS_SIMULATION_ID,
        models_infotaxis.Trial.geom_config_id == gc.id).first()

    x_idxs = trial.timepoint_field(session_infotaxis, 'xidx')
    y_idxs = trial.timepoint_field(session_infotaxis, 'yidx')
    z_idxs = trial.timepoint_field(session_infotaxis, 'zidx')
    cs_infotaxis = trial.timepoint_field(session_infotaxis, 'odor')

    x_idxs = x_idxs[:end_tp_info]
    y_idxs = y_idxs[:end_tp_info]
    z_idxs = z_idxs[:end_tp_info]
    cs_infotaxis = cs_infotaxis[:end_tp_info]

    # convert to positions
    sim = trial.simulation
    env = sim.env

    xs_infotaxis = []
    ys_infotaxis = []
    zs_infotaxis = []

    for x_idx, y_idx, z_idx in zip(x_idxs, y_idxs, z_idxs):
        x, y, z = env.pos_from_idx([x_idx, y_idx, z_idx])
        xs_infotaxis.append(x)
        ys_infotaxis.append(y)
        zs_infotaxis.append(z)

    xs_infotaxis = np.array(xs_infotaxis)
    ys_infotaxis = np.array(ys_infotaxis)
    zs_infotaxis = np.array(zs_infotaxis)

    traj_dict_infotaxis = {
        'title': 'infotaxis',
        'xs': xs_infotaxis,
        'ys': ys_infotaxis,
        'zs': zs_infotaxis,
        'cs': cs_infotaxis,
    }

    ## MAKE PLOTS
    fig = plt.figure(figsize=FIG_SIZE, tight_layout=True)
    axs = [fig.add_subplot(4, 1, ctr+1, projection='3d') for ctr in range(4)]

    # plot all trajectories
    traj_dicts = [
        traj_dict_real,
        traj_dict_surge,
        traj_dict_centerline,
        traj_dict_infotaxis
    ]

    for traj_dict, ax in zip(traj_dicts, axs):

        # overlay plume cylinder
        CYL_MEAN_Y = PLUME_PARAMS_DICT[EXPT_ID]['ymean']
        CYL_MEAN_Z = PLUME_PARAMS_DICT[EXPT_ID]['zmean']

        CYL_SCALE_Y = CYL_STDS * PLUME_PARAMS_DICT[EXPT_ID]['ystd']
        CYL_SCALE_Z = CYL_STDS * PLUME_PARAMS_DICT[EXPT_ID]['zstd']

        MAX_CONC = PLUME_PARAMS_DICT[EXPT_ID]['max_conc']

        y = np.linspace(-1, 1, 100, endpoint=True)
        x = np.linspace(-0.3, 1, 5, endpoint=True)
        yy, xx = np.meshgrid(y, x)
        zz = np.sqrt(1 - yy ** 2)

        yy = CYL_SCALE_Y * yy + CYL_MEAN_Y
        zz_top = CYL_SCALE_Z * zz + CYL_MEAN_Z
        zz_bottom = -CYL_SCALE_Z * zz + CYL_MEAN_Z
        rstride = 20
        cstride = 10

        ax.plot_surface(
            xx, yy, zz_top, lw=0,
            color=CYL_COLOR, alpha=CYL_ALPHA,
            rstride=rstride, cstride=cstride)

        ax.plot_surface(
            xx, yy, zz_bottom, lw=0,
            color=CYL_COLOR, alpha=CYL_ALPHA,
            rstride=rstride, cstride=cstride)

        # plot trajectory
        xs_ = traj_dict['xs']
        ys_ = traj_dict['ys']
        zs_ = traj_dict['zs']
        cs_ = traj_dict['cs']

        # show trajectory
        ax.plot(xs_, ys_, zs_, color='k', lw=3, zorder=1)
        # overlay concentrations
        ax.scatter(
            xs_, ys_, zs_, c=cs_, s=SCATTER_SIZE, vmin=0, vmax=MAX_CONC/2,
            cmap=cmx.hot, lw=0, alpha=1, zorder=2)
        # mark start
        ax.scatter(
            xs_[0], ys_[0], zs_[0], s=400, marker='*', lw=0, c='g', zorder=3)

        ax.set_xlim(-0.3, 1)
        ax.set_ylim(-0.15, 0.15)
        ax.set_zlim(-0.15, 0.15)

        ax.set_xticks([-0.3, 1.])
        ax.set_yticks([-0.15, 0.15])
        ax.set_zticks([-0.15, 0.15])

        ax.set_xticklabels([-30, 100])
        ax.set_yticklabels([-15, 15])
        ax.set_zticklabels([-15, 15])

        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        ax.set_zlabel('z (cm)')

        ax.set_title(traj_dict['title'])

    for ax in axs:
        set_fontsize(ax, FONT_SIZE)

    return fig


def infotaxis_average_dt(INFOTAXIS_SIM_IDS):
    """
    Print out the average DT used in infotaxis trajectories.
    """
    from db_api.infotaxis import models as models_infotaxis
    from db_api.infotaxis.connect import session as session_infotaxis

    for sim_id in INFOTAXIS_SIM_IDS:
        print('Simulation: {}'.format(sim_id))
        trials = session_infotaxis.query(models_infotaxis.Trial).filter_by(
            simulation_id=sim_id).all()
        geom_configs = [trial.geom_config for trial in trials]
        avg_dts = [
            geom_config.extension_real_trajectory.avg_dt
            for geom_config in geom_configs
        ]
        print('Average DT = {}'.format(np.mean(avg_dts)))


def hybrid_model_history_dependence(
        SEED, N_TRIALS, EMPIRICAL_LATE_EARLY_DIFF, SURGE_CAST_FILE,
        INFOTAXIS_CG_ID, INFOTAXIS_DT,
        X_0_MIN_SIM_HISTORY, X_0_MAX_SIM_HISTORY,
        H_0_MIN, H_0_MAX, TS_BEFORE_SIM, TS_AFTER_SIM, HEADING_SMOOTHING_SIM,
        EARLY_LESS_THAN, T_AVG_DIFF_START, T_AVG_DIFF_END, N_XS):
    """
    Show how close the model history dependence matches the empirical history
    dependence as a function of the surge-cast vs. infotaxis mixture percent.
    """
    np.random.seed(SEED)
    random.seed(SEED)
    from db_api.infotaxis import models as models_infotaxis
    from db_api.infotaxis.connect import session as session_infotaxis

    # load and sort surge-cast crossings
    data_sc = np.load(SURGE_CAST_FILE)[0]
    data_sc['crossings_sorted'] = {}
    t_steps = np.arange(-data_sc['ts_before'], data_sc['ts_after'])
    ts = t_steps * DT

    for cn, crossing in data_sc['crossings']:
        if cn not in data_sc['crossings_sorted']:
            data_sc['crossings_sorted'][cn] = []
        data_sc['crossings_sorted'][cn].append(crossing)

    # load infotaxis crossings
    data_it = {
        'crossings_sorted': {},
    }

    # get headings from infotaxis plume crossings
    crossings_it_all = list(session_infotaxis.query(
        models_infotaxis.Crossing).filter_by(
        crossing_group_id=INFOTAXIS_CG_ID).all())

    for crossing in crossings_it_all:

        # skip this crossing if it doesn't meet our inclusion criteria
        x_0 = crossing.feature_set_basic.position_x_peak
        h_0 = crossing.feature_set_basic.heading_xyz_peak

        if not (X_0_MIN_SIM_HISTORY <= x_0 <= X_0_MAX_SIM_HISTORY):
            continue
        if not (H_0_MIN <= h_0 <= H_0_MAX):
            continue

        # store crossing heading
        temp = crossing.timepoint_field(
            session_infotaxis, 'hxyz', -TS_BEFORE_SIM, TS_AFTER_SIM - 1,
            'peak', 'peak', nan_pad=True)

        temp[~np.isnan(temp)] = gaussian_filter1d(
            temp[~np.isnan(temp)], HEADING_SMOOTHING_SIM)

        t_temp = np.arange(-TS_BEFORE_SIM, TS_AFTER_SIM) * INFOTAXIS_DT

        # convert crossing to be on same timescale as surge-cast crossings
        crossing_r = np.nan * np.zeros(ts.shape)
        # identify nans for removal during resampling
        temp_mask = ~np.isnan(temp)
        # resample non-nan portion of crossing
        t_start = t_temp[temp_mask][0]
        t_zero = t_temp[TS_BEFORE_SIM]
        t_end = t_temp[temp_mask][-1]
        n_before = int(np.ceil((t_zero - t_start) / DT))
        n_after = int(np.floor((t_end - t_zero) / DT))
        crossing_r_ = resample(temp[temp_mask], n_before + n_after)

        # insert resampled crossing into correct position in crossing_r
        t_start_r = data_sc['ts_before'] - n_before
        if t_start_r < 0:
            crossing_r_ = crossing_r_[-t_start_r:]
            t_start_r = 0
        if len(crossing_r_) > len(crossing_r) - t_start_r:
            crossing_r_ = crossing_r_[:len(crossing_r) - t_start_r]

        crossing_r[t_start_r:t_start_r + len(crossing_r_)] = crossing_r_

        #  subtract initial heading
        if ~np.isnan(crossing_r[data_sc['ts_before']]):
            crossing_r -= crossing_r[data_sc['ts_before']]
        else:
            crossing_r -= crossing_r[data_sc['ts_before'] - 1]
        # make sure crossing_r is not all nans for some reason
        assert not np.all(np.isnan(crossing_r))

        # store according to its crossing number
        cn = crossing.crossing_number
        if cn not in data_it['crossings_sorted']:
            data_it['crossings_sorted'][cn] = []
        data_it['crossings_sorted'][cn].append(crossing_r.copy())

    # loop over trials
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), tight_layout=True)
    x_bests = []
    no_plot_yet = True
    for trial_ctr in range(N_TRIALS):

        # pair each infotaxis crossing with a surge-cast crossing of the same
        # crossing number
        crossing_pairs = {}
        for key in data_it['crossings_sorted']:
            if key not in data_sc['crossings_sorted']:
                continue
            crossing_pairs[key] = []
            for crossing_it in data_it['crossings_sorted'][key]:
                crossing_sc = choice(data_sc['crossings_sorted'][key])
                crossing_pairs[key].append((crossing_sc.copy(), crossing_it.copy()))

        # loop over xs and create set of hybrid crossings for each x
        xs = np.linspace(0, 1, N_XS)
        model_mean_diff_t_avgs = []
        t_avg_mask = (T_AVG_DIFF_START <= ts) * (ts < T_AVG_DIFF_END)
        t_avg_mask = t_avg_mask.astype(bool)

        abs_data_model_diff_best = np.inf

        for x in xs:
            # create hybrid crossing groups
            crossings_sorted_hybrid = {}
            for key, crossing_pairs_ in crossing_pairs.items():
                crossings_sorted_hybrid[key] = []
                for crossing_sc, crossing_it in crossing_pairs_:
                    crossing_hybrid = (x * crossing_sc) + ((1-x) * crossing_it)
                    crossings_sorted_hybrid[key].append(crossing_hybrid)

            # separate into early vs. late crossings
            crossings_early = []
            crossings_late = []
            for key, crossing_set in crossings_sorted_hybrid.items():
                if key < EARLY_LESS_THAN:
                    crossings_early.extend(crossing_set)
                else:
                    crossings_late.extend(crossing_set)

            # calculate history dependence
            crossings_early = np.array(crossings_early)
            crossings_late = np.array(crossings_late)

            mean_early = np.nanmean(crossings_early, axis=0)
            mean_late = np.nanmean(crossings_late, axis=0)

            mean_diff = mean_late - mean_early
            mean_diff_t_avg = mean_diff[t_avg_mask].mean()

            model_mean_diff_t_avgs.append(mean_diff_t_avg)

            abs_data_model_diff = np.abs(mean_diff_t_avg - EMPIRICAL_LATE_EARLY_DIFF)
            if abs_data_model_diff < abs_data_model_diff_best:
                x_best = x
                abs_data_model_diff_best = abs_data_model_diff
                crossings_early_best = crossings_early
                crossings_late_best = crossings_late

        model_mean_diff_t_avgs = np.array(model_mean_diff_t_avgs)
        x_bests.append(x_best)

        if np.abs(x_best - 0.8) < 0.001 and no_plot_yet:
            no_plot_yet = False
            # plot the data-model difference as a function of x
            axs[0].plot(100*xs, model_mean_diff_t_avgs, color='r', lw=2)
            axs[0].axhline(EMPIRICAL_LATE_EARLY_DIFF, color='k', lw=2)
            axs[0].set_xlabel('% surge-cast')
            axs[0].set_ylabel('time-averaged\nlate mean minus early mean')
            axs[0].legend(['hybrid model', 'data'])

            # plot early and late crossing groups
            mean_early = np.nanmean(crossings_early_best, axis=0)
            mean_late = np.nanmean(crossings_late_best, axis=0)
            sem_early = stats.nansem(crossings_early_best, axis=0)
            sem_late = stats.nansem(crossings_late_best, axis=0)

            axs[1].plot(ts, mean_early, color='b', lw=2)
            axs[1].plot(ts, mean_late, color='g', lw=2)
            axs[1].fill_between(
                ts, mean_early-sem_early, mean_early+sem_early,
                color='b', alpha=0.2)
            axs[1].fill_between(
                ts, mean_late-sem_late, mean_late+sem_late,
                color='g', alpha=0.2)
            axs[1].set_xlabel('time since crossing (s)')
            axs[1].set_ylabel('change in heading (deg)')
            axs[1].set_title(
                ('history dependence for \n{0:.1f}% surge-cast, '
                 '{1:.1f}% infotaxis'.format(100*x_best, 100*(1-x_best))))

    axs[2].hist(100 * np.array(x_bests), bins=20)
    axs[2].set_xlabel('best % surge-cast')
    axs[2].set_ylabel('trials')
    axs[2].set_title('mean = {}'.format(np.mean(x_bests)))

    for ax in axs:
        set_fontsize(ax, 16)

    return fig



def infotaxis_position_distribution(
        HEAT_MAP_EXPT_ID, HEAT_MAP_SIM_ID, N_HEAT_MAP_TRAJS, X_BINS, Y_BINS,
        FIG_SIZE, FONT_SIZE, EXPT_LABELS, EXPT_COLORS, SIM_LABELS):
    """
    Show infotaxis-generated trajectories alongside empirical trajectories. Show wind-speed
    dependence and history dependence.
    """

    from db_api.infotaxis import models as models_infotaxis
    from db_api.infotaxis.connect import session as session_infotaxis

    # get heatmaps
    if N_HEAT_MAP_TRAJS:
        trajs_expt = session.query(models.Trajectory).\
            filter_by(experiment_id=HEAT_MAP_EXPT_ID, odor_state='on').limit(N_HEAT_MAP_TRAJS)
        trials_sim = session_infotaxis.query(models_infotaxis.Trial).\
            filter_by(simulation_id=HEAT_MAP_SIM_ID).limit(N_HEAT_MAP_TRAJS)

    else:
        trajs_expt = session.query(models.Trajectory).\
            filter_by(experiment_id=HEAT_MAP_EXPT_ID, odor_state='on')
        trials_sim = session_infotaxis.query(models_infotaxis.Trial).\
            filter_by(simulation_id=HEAT_MAP_SIM_ID)

    expt_xs = []
    expt_ys = []

    sim_xs = []
    sim_ys = []

    for traj in trajs_expt:
        expt_xs.append(traj.timepoint_field(session, 'position_x'))
        expt_ys.append(traj.timepoint_field(session, 'position_y'))

    for trial in trials_sim:
        sim_xs.append(trial.timepoint_field(session_infotaxis, 'xidx'))
        sim_ys.append(trial.timepoint_field(session_infotaxis, 'yidx'))

    expt_xs = np.concatenate(expt_xs)
    expt_ys = np.concatenate(expt_ys)

    sim_xs = np.concatenate(sim_xs) * 0.02 - 0.3
    sim_ys = np.concatenate(sim_ys) * 0.02 - 0.15

    ## MAKE PLOTS
    fig, axs = plt.subplots(2, 1, figsize=FIG_SIZE, tight_layout=True)

    # plot heat maps
    axs[0].hist2d(expt_xs, expt_ys, bins=(X_BINS, Y_BINS))
    axs[1].hist2d(sim_xs, sim_ys, bins=(X_BINS, Y_BINS))

    axs[0].set_ylabel('y (m)')
    axs[0].set_xlabel('x (m)')
    axs[1].set_ylabel('y (m)')
    axs[1].set_xlabel('x (m)')

    axs[0].set_title('experimental')
    axs[1].set_title('infotaxis simulation')

    for ax in axs:
        set_fontsize(ax, FONT_SIZE)

    return fig


def infotaxis_wind_speed_dependence(
        WIND_TUNNEL_CG_IDS, INFOTAXIS_WIND_SPEED_CG_IDS, MAX_CROSSINGS,
        X_0_MIN, X_0_MAX, H_0_MIN, H_0_MAX,
        X_0_MIN_SIM, X_0_MAX_SIM,
        T_BEFORE_EXPT, T_AFTER_EXPT, TS_BEFORE_SIM, TS_AFTER_SIM, HEADING_SMOOTHING_SIM,
        FIG_SIZE, FONT_SIZE, EXPT_LABELS, EXPT_COLORS, SIM_LABELS):
    """
    Show infotaxis-generated trajectories alongside empirical trajectories. Show wind-speed
    dependence and history dependence.
    """

    from db_api.infotaxis import models as models_infotaxis
    from db_api.infotaxis.connect import session as session_infotaxis

    ts_before_expt = int(round(T_BEFORE_EXPT / DT))
    ts_after_expt = int(round(T_AFTER_EXPT / DT))

    headings = {}

    # get headings for wind tunnel plume crossings
    headings['wind_tunnel'] = {}

    for cg_id in WIND_TUNNEL_CG_IDS:
        crossings_all = session.query(models.Crossing).filter_by(crossing_group_id=cg_id).all()
        headings['wind_tunnel'][cg_id] = []

        cr_ctr = 0

        for crossing in crossings_all:
            if cr_ctr >= MAX_CROSSINGS:
                break

            # skip this crossing if it doesn't meet our inclusion criteria
            x_0 = crossing.feature_set_basic.position_x_peak
            h_0 = crossing.feature_set_basic.heading_xyz_peak

            if not (X_0_MIN <= x_0 <= X_0_MAX):
                continue
            if not (H_0_MIN <= h_0 <= H_0_MAX):
                continue

            # store crossing heading
            temp = crossing.timepoint_field(
                session, 'heading_xyz', -ts_before_expt, ts_after_expt - 1,
                'peak', 'peak', nan_pad=True)

            # subtract initial heading
            temp -= temp[ts_before_expt]
            headings['wind_tunnel'][cg_id].append(temp)

            cr_ctr += 1

        headings['wind_tunnel'][cg_id] = np.array(headings['wind_tunnel'][cg_id])

    # get headings from infotaxis plume crossings
    headings['infotaxis'] = {}

    for cg_id in INFOTAXIS_WIND_SPEED_CG_IDS:

        crossings_all = list(session_infotaxis.query(models_infotaxis.Crossing).filter_by(
            crossing_group_id=cg_id).all())

        print('{} crossings for infotaxis crossing group: "{}"'.format(
            len(crossings_all), cg_id))

        headings['infotaxis'][cg_id] = []

        cr_ctr = 0

        for crossing in crossings_all:
            if cr_ctr >= MAX_CROSSINGS:
                break

            # skip this crossing if it doesn't meet our inclusion criteria
            x_0 = crossing.feature_set_basic.position_x_peak
            h_0 = crossing.feature_set_basic.heading_xyz_peak

            if not (X_0_MIN_SIM <= x_0 <= X_0_MAX_SIM):
                continue
            if not (H_0_MIN <= h_0 <= H_0_MAX):
                continue

            # store crossing heading
            temp = crossing.timepoint_field(
                session_infotaxis, 'hxyz', -TS_BEFORE_SIM, TS_AFTER_SIM - 1,
                'peak', 'peak', nan_pad=True)

            temp[~np.isnan(temp)] = gaussian_filter1d(
                temp[~np.isnan(temp)], HEADING_SMOOTHING_SIM)

            # subtract initial heading and store result
            temp -= temp[TS_BEFORE_SIM]
            headings['infotaxis'][cg_id].append(temp)

            cr_ctr += 1

        headings['infotaxis'][cg_id] = np.array(headings['infotaxis'][cg_id])

    ## MAKE PLOTS
    fig, axs = plt.subplots(1, 2, figsize=FIG_SIZE, tight_layout=True)

    # plot wind-speed dependence of wind tunnel trajectories
    t = np.arange(-ts_before_expt, ts_after_expt) * DT

    handles = []

    for cg_id in WIND_TUNNEL_CG_IDS:

        label = EXPT_LABELS[cg_id]
        color = EXPT_COLORS[cg_id]

        headings_mean = np.nanmean(headings['wind_tunnel'][cg_id], axis=0)
        headings_sem = stats.nansem(headings['wind_tunnel'][cg_id], axis=0)

        # plot mean and sem
        handles.append(
            axs[0].plot(t, headings_mean, lw=3, color=color, zorder=1, label=label)[0])
        axs[0].fill_between(
            t, headings_mean - headings_sem, headings_mean + headings_sem,
            color=color, alpha=0.2)

    axs[0].set_xlabel('time since crossing (s)')
    axs[0].set_ylabel('$\Delta$ heading (degrees)')
    axs[0].set_title('empirical (fly in ethanol)')

    axs[0].legend(handles=handles, loc='best')

    t = np.arange(-TS_BEFORE_SIM, TS_AFTER_SIM)

    for cg_id, wt_cg_id in zip(INFOTAXIS_WIND_SPEED_CG_IDS, WIND_TUNNEL_CG_IDS):
        label = EXPT_LABELS[wt_cg_id]
        color = EXPT_COLORS[wt_cg_id]

        headings_mean = np.nanmean(headings['infotaxis'][cg_id], axis=0)
        headings_sem = stats.nansem(headings['infotaxis'][cg_id], axis=0)

        # plot mean and sem
        axs[1].plot(t, headings_mean, lw=3, color=color, zorder=1, label=label)
        axs[1].fill_between(
            t, headings_mean - headings_sem, headings_mean + headings_sem,
            color=color, alpha=0.2)

    axs[1].set_xlabel('time steps since crossing (s)')
    axs[1].set_title('infotaxis')

    for ax in axs:
        set_fontsize(ax, FONT_SIZE)

    return fig



