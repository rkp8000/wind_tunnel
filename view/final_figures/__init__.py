from __future__ import division, print_function
from copy import deepcopy
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pprint import pprint
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ks_2samp

from db_api import models
from db_api.connect import session

from axis_tools import set_fontsize
import simple_models
import stats
import time_series_classifier as tsc

from experimental_constants import DT, PLUME_PARAMS_DICT


def example_traj_and_crossings(
        SEED,
        EXPT_ID,
        TRAJ_NUMBER,
        TRAJ_START_TP, TRAJ_END_TP,
        CROSSING_GROUP,
        N_CROSSINGS,
        X_0_MIN, X_0_MAX, H_0_MIN, H_0_MAX,
        MIN_PEAK_CONC,
        TS_BEFORE_3D, TS_AFTER_3D,
        TS_BEFORE_HEADING, TS_AFTER_HEADING,
        FIG_SIZE,
        SCATTER_SIZE,
        CYL_STDS, CYL_COLOR, CYL_ALPHA,
        EXPT_LABEL,
        FONT_SIZE):
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
        crossing_group_id=CROSSING_GROUP).filter(models.Crossing.max_odor > MIN_PEAK_CONC).all()
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

    axs[0].scatter(
        x_traj, y_traj, z_traj, c=c_traj, s=SCATTER_SIZE,
        vmin=0, vmax=MAX_CONC/2, cmap=cmx.hot, lw=0, alpha=1)

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
    ts_models = {cg_id: int(round(t_model / DT)) for cg_id, t_model in T_MODELS.items()}

    data = {cg_id: None for cg_id in CROSSING_GROUP_IDS}

    for cg_id in CROSSING_GROUP_IDS:

        # get crossing group and crossings

        crossing_group = session.query(models.CrossingGroup).filter_by(id=cg_id).first()
        crossings_all = session.query(models.Crossing).filter_by(crossing_group=crossing_group)

        # get all initial headings, initial xs, peak concentrations, and heading time-series

        x_0s = []
        h_0s = []
        c_maxs = []
        headings = []

        for crossing in crossings_all:

            # throw away crossings that do not meet trigger criteria

            position_x = getattr(crossing.feature_set_basic, 'position_x_{}'.format('peak'))

            if not (X_0_MIN <= position_x <= X_0_MAX):

                continue

            heading_xyz = getattr(crossing.feature_set_basic, 'heading_xyz_{}'.format('peak'))

            if not (H_0_MIN <= heading_xyz <= H_0_MAX):

                continue

            c_maxs.append(crossing.max_odor * conversion_factor)
            x_0s.append(position_x)
            h_0s.append(heading_xyz)

            temp = crossing.timepoint_field(
                session, 'heading_xyz', -ts_before, ts_after - 1,
                'peak', 'peak', nan_pad=True)

            headings.append(temp)

        x_0s = np.array(x_0s)
        h_0s = np.array(h_0s)
        c_maxs = np.array(c_maxs)
        headings = np.array(headings)

        partial_corrs = np.nan * np.ones((headings.shape[1],), dtype=float)
        p_vals = np.nan * np.ones((headings.shape[1],), dtype=float)
        lbs = np.nan * np.ones((headings.shape[1],), dtype=float)
        ubs = np.nan * np.ones((headings.shape[1],), dtype=float)
        ns = np.nan * np.ones((headings.shape[1],), dtype=float)

        # loop through all time steps

        for ts in range(headings.shape[1]):

            headings_this_tp = headings[:, ts]

            if ts == (ts_models[cg_id] + ts_before):

                model_headings = headings_this_tp.copy()

            # create not-nan mask

            mask = ~np.isnan(headings_this_tp)
            ns[ts] = mask.sum()

            # get partial correlations using all not-nan values

            r, p, lb, ub = stats.pearsonr_partial_with_confidence(
                c_maxs[mask], headings_this_tp[mask],
                [x_0s[mask], h_0s[mask]])

            partial_corrs[ts] = r
            p_vals[ts] = p
            lbs[ts] = lb
            ubs[ts] = ub

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
        }

    ## MAKE PLOT OF PARTIAL CORRELATIONS

    fig, axs = plt.figure(figsize=FIG_SIZE, facecolor='white', tight_layout=True), []

    axs.append(fig.add_subplot(2, 1, 1))
    axs.append(axs[0].twinx())

    axs[1].axhline(0.05)

    t = np.arange(-ts_before, ts_after) * DT
    t[ts_before] = np.nan

    handles = []

    for cg_id in CROSSING_GROUP_IDS:

        color = CROSSING_GROUP_COLORS[cg_id]
        label = CROSSING_GROUP_LABELS[cg_id]

        # show partial correlation and confidence

        handle = axs[0].plot(
            t, data[cg_id]['partial_corrs'], color=color, lw=2, ls='-', label=label)[0]
        axs[0].fill_between(
            t, data[cg_id]['lbs'], data[cg_id]['ubs'], color=color, alpha=0.2)

        handles.append(handle)

        # show p-values

        axs[1].plot(t[t > 0], data[cg_id]['p_vals'][t > 0], color=color, lw=2, ls='--')

    axs[0].axhline(0, color='gray', ls='--')
    axs[0].set_xlim(-T_BEFORE, T_AFTER)

    axs[0].set_xlabel('time of heading measurement\nsince crossing (s)')
    axs[0].set_ylabel('heading-concentration\npartial correlation')
    axs[0].legend(handles=handles, loc='upper left')

    axs[1].set_ylim(0, 0.2)

    axs[1].set_ylabel('p-value (dashed lines)')

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

        n = len(hs)

        rho = stats.pearsonr_partial_with_confidence(c_maxs, hs, [x_0s, h_0s])[0]

        binary_model = simple_models.ThresholdLinearHeadingConcModel(
            include_c_max_coefficient=False)

        binary_model.brute_force_fit(hs=hs, c_maxs=c_maxs, x_0s=x_0s, h_0s=h_0s)

        hs_predicted_binary = binary_model.predict(c_maxs=c_maxs, x_0s=x_0s, h_0s=h_0s)

        rss_binary = np.sum((hs - hs_predicted_binary) ** 2)

        threshold_linear_model = simple_models.ThresholdLinearHeadingConcModel(
            include_c_max_coefficient=True)

        threshold_linear_model.brute_force_fit(hs=hs, c_maxs=c_maxs, x_0s=x_0s, h_0s=h_0s)

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
    axs[-1].set_ylabel('heading at {} s\n since crossing'.format(T_MODELS[CROSSING_GROUP_EXAMPLE_ID]))
    axs[-1].set_title('heading-concentration relationship for {}'.format(CROSSING_GROUP_LABELS[CROSSING_GROUP_EXAMPLE_ID]))

    for ax in axs:

        set_fontsize(ax, FONT_SIZE)

    return fig


def early_vs_late_heading_timecourse(
        CROSSING_GROUP_IDS, CROSSING_GROUP_LABELS,
        X_0_MIN, X_0_MAX, H_0_MIN, H_0_MAX,
        MAX_CROSSINGS_EARLY, SUBTRACT_INITIAL_HEADING,
        T_BEFORE, T_AFTER,
        AX_SIZE, AX_GRID, EARLY_LATE_COLORS, ALPHA,
        P_VAL_COLOR, P_VAL_Y_LIM, LEGEND_CROSSING_GROUP_ID,
        X_0_BINS, FONT_SIZE):
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

        # get crossing group
        crossing_group = session.query(models.CrossingGroup).filter_by(id=cg_id).first()

        # get early and late crossings
        crossings_dict = {}
        crossings_all = session.query(models.Crossing).filter_by(crossing_group=crossing_group)
        crossings_dict['early'] = crossings_all.filter(models.Crossing.crossing_number <= MAX_CROSSINGS_EARLY)
        crossings_dict['late'] = crossings_all.filter(models.Crossing.crossing_number > MAX_CROSSINGS_EARLY)

        x_0s_dict[cg_id] = {}
        headings_dict[cg_id] = {}

        for label in ['early', 'late']:

            x_0s = []
            headings = []

            # get all initial headings, initial xs, peak concentrations, and heading time-series
            for crossing in crossings_dict[label]:

                # throw away crossings that do not meet trigger criteria
                x_0 = getattr(crossing.feature_set_basic, 'position_x_{}'.format('peak'))
                h_0 = getattr(crossing.feature_set_basic, 'heading_xyz_{}'.format('peak'))

                if not (X_0_MIN <= x_0 <= X_0_MAX):
                    continue

                if not (H_0_MIN <= h_0 <= H_0_MAX):
                    continue

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

        # loop through all time points and calculate p-value (ks-test) between early and late
        p_vals = []

        for t_step in range(ts_before + ts_after):

            early_with_nans = headings_dict[cg_id]['early'][:, t_step]
            late_with_nans = headings_dict[cg_id]['late'][:, t_step]

            early_no_nans = early_with_nans[~np.isnan(early_with_nans)]
            late_no_nans = late_with_nans[~np.isnan(late_with_nans)]

            p_vals.append(ks_2samp(early_no_nans, late_no_nans)[1])

        p_vals_dict[cg_id] = p_vals

    ## MAKE PLOTS
    t = np.arange(-ts_before, ts_after) * DT

    # history-dependence
    fig_size = (AX_SIZE[0] * AX_GRID[1], AX_SIZE[1] * AX_GRID[0])
    fig_0, axs_0 = plt.subplots(*AX_GRID, figsize=fig_size, tight_layout=True)

    for cg_id, ax in zip(CROSSING_GROUP_IDS, axs_0.flat):

        # get mean and sem of headings for early and late groups
        handles = []

        for label, color in EARLY_LATE_COLORS.items():

            headings_mean = np.nanmean(headings_dict[cg_id][label], axis=0)
            headings_sem = stats.nansem(headings_dict[cg_id][label], axis=0)

            handles.append(ax.plot(t, headings_mean, color=color, lw=2, label=label, zorder=1)[0])
            ax.fill_between(
                t, headings_mean - headings_sem, headings_mean + headings_sem,
                color=color, alpha=ALPHA, zorder=1)

        ax.set_xlabel('time since crossing (s)')

        if SUBTRACT_INITIAL_HEADING:
            ax.set_ylabel('$\Delta$ heading (deg.)')
        else:
            ax.set_ylabel('heading (deg.)')

        ax.set_title(CROSSING_GROUP_LABELS[cg_id])

        if cg_id == LEGEND_CROSSING_GROUP_ID:
            ax.legend(handles=handles, loc='upper right')

        set_fontsize(ax, FONT_SIZE)

        # plot p-value
        ax_twin = ax.twinx()

        ax_twin.plot(t, p_vals_dict[cg_id], color=P_VAL_COLOR, lw=2, ls='--', zorder=0)
        ax_twin.axhline(0.05, ls='-', lw=2, color='gray')

        ax_twin.set_ylim(*P_VAL_Y_LIM)
        ax_twin.set_ylabel('p-value (KS test)', fontsize=FONT_SIZE)

        set_fontsize(ax_twin, FONT_SIZE)

    # position histograms
    bincs = 0.5 * (X_0_BINS[:-1] + X_0_BINS[1:])
    fig_1, axs_1 = plt.subplots(*AX_GRID, figsize=fig_size, tight_layout=True)

    for cg_id, ax in zip(CROSSING_GROUP_IDS, axs_1.flat):

        # create early and late histograms
        handles = []

        for label, color in EARLY_LATE_COLORS.items():

            probs = np.histogram(x_0s_dict[cg_id][label], bins=X_0_BINS, normed=True)[0]
            handles.append(ax.plot(100*bincs, probs, lw=2, color=color, label=label)[0])

        p_val = ks_2samp(x_0s_dict[cg_id]['early'], x_0s_dict[cg_id]['late'])[1]
        x_0_mean_diff = x_0s_dict[cg_id]['late'].mean() - x_0s_dict[cg_id]['early'].mean()

        ax.set_xlabel(r'$x_0$ (cm)')
        ax.set_ylabel('proportion of\ncrossings')

        title = '{0} ($\Delta mean(x_0)$ = {1:10.2} cm) \n(p = {2:10.5f} [KS test])'.format(
            CROSSING_GROUP_LABELS[cg_id], 100 * x_0_mean_diff, p_val)

        ax.set_title(title)
        ax.legend(handles=handles, loc='best')
        set_fontsize(ax, FONT_SIZE)

    return fig_0, fig_1


def infotaxis_analysis(
        WIND_TUNNEL_CG_IDS,
        INFOTAXIS_WIND_SPEED_CG_IDS,
        MAX_CROSSINGS,
        INFOTAXIS_HISTORY_DEPENDENCE_CG_IDS,
        MAX_CROSSINGS_EARLY,
        X_0_MIN, X_0_MAX, H_0_MIN, H_0_MAX,
        X_0_MIN_SIM, X_0_MAX_SIM,
        X_0_MIN_SIM_HISTORY, X_0_MAX_SIM_HISTORY,
        T_BEFORE_EXPT, T_AFTER_EXPT,
        TS_BEFORE_SIM, TS_AFTER_SIM, HEADING_SMOOTHING_SIM,
        HEAT_MAP_EXPT_ID, HEAT_MAP_SIM_ID,
        N_HEAT_MAP_TRAJS,
        X_BINS, Y_BINS,
        FIG_SIZE, FONT_SIZE,
        EXPT_LABELS,
        EXPT_COLORS,
        SIM_LABELS):
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

    # get history dependences for infotaxis simulations

    headings['it_hist_dependence'] = {}

    for cg_id in INFOTAXIS_HISTORY_DEPENDENCE_CG_IDS:

        crossings_all = list(session_infotaxis.query(models_infotaxis.Crossing).filter_by(
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

                raise Exception('crossing number is not early or late for crossing {}'.format(
                    crossing.id))

            cr_ctr += 1

    headings['it_hist_dependence'][cg_id]['early'] = np.array(
        headings['it_hist_dependence'][cg_id]['early'])

    headings['it_hist_dependence'][cg_id]['late'] = np.array(
        headings['it_hist_dependence'][cg_id]['late'])

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

    fig, axs = plt.figure(figsize=FIG_SIZE, tight_layout=True), []

    axs.append(fig.add_subplot(4, 3, 1))
    axs.append(fig.add_subplot(4, 3, 2, sharey=axs[0]))

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

    axs[0].set_xlabel('time since odor peak (s)')
    axs[0].set_ylabel('$\Delta$ heading (degrees)')
    axs[0].set_title('experimental data\n(wind speed comparison)')

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

    axs[1].set_xlabel('time steps since odor peak (s)')
    axs[1].set_title('infotaxis simulations\n(wind speed comparison)')

    # add axes for infotaxis history dependence and make plots

    [axs.append(fig.add_subplot(4, 3, 3 + ctr)) for ctr in range(4)]

    for (ax, cg_id) in zip(axs[-4:], INFOTAXIS_HISTORY_DEPENDENCE_CG_IDS):

        mean_early = np.nanmean(headings['it_hist_dependence'][cg_id]['early'], axis=0)
        sem_early = stats.nansem(headings['it_hist_dependence'][cg_id]['early'], axis=0)

        mean_late = np.nanmean(headings['it_hist_dependence'][cg_id]['late'], axis=0)
        sem_late = stats.nansem(headings['it_hist_dependence'][cg_id]['late'], axis=0)

        # plot means and stds

        try:

            handle_early = ax.plot(t, mean_early, lw=3, color='b', zorder=0, label='early')[0]
            ax.fill_between(
                t, mean_early - sem_early, mean_early + sem_early,
                color='b', alpha=0.2)

        except:

            pass

        try:

            handle_late = ax.plot(t, mean_late, lw=3, color='g', zorder=0, label='late')[0]
            ax.fill_between(
                t, mean_late - sem_late, mean_late + sem_late,
                color='g', alpha=0.2)

        except:

            pass

        ax.set_xlabel('time steps since odor peak (s)')
        ax.set_title(SIM_LABELS[cg_id])

        try:

            ax.legend(handles=[handle_early, handle_late])

        except:

            pass

    axs[3].set_ylabel('$\Delta$ heading (degrees)')

    # plot heat maps

    axs.append(fig.add_subplot(4, 1, 3))
    axs.append(fig.add_subplot(4, 1, 4))

    axs[6].hist2d(expt_xs, expt_ys, bins=(X_BINS, Y_BINS))
    axs[7].hist2d(sim_xs, sim_ys, bins=(X_BINS, Y_BINS))

    axs[6].set_ylabel('y (m)')
    axs[7].set_ylabel('y (m)')
    axs[7].set_xlabel('x (m)')

    axs[6].set_title('experimental data (fly 0.4 m/s)')
    axs[7].set_title('infotaxis simulation')

    for ax in axs:

        set_fontsize(ax, FONT_SIZE)

    return fig


def classify_trajectories(
        SEED,
        EXPT_IDS,
        CLASSIFIERS,
        MIN_INTEGRATED_ODORS, MIN_MAX_ODORS,
        N_TRAJS_PER_CLASS, PROPORTION_TRAIN, N_TRIALS,
        BINS, AX_SIZE, EXPT_LABELS, FONT_SIZE):
    """
    Show classification accuracy when classifying whether insect is engaged in odor-tracking
    or not.
    """

    np.random.seed(SEED)

    # loop through experiments

    n_traj_pairs = {}
    train_accuracies = {}
    test_accuracies = {}

    for e_ctr, expt_id in enumerate(EXPT_IDS):

        print('Experiment: {}'.format(expt_id))

        # get all trajectories that meet criteria

        trajs = {'on': [], 'none': []}

        for odor_state in trajs.keys():

            traj_array = list(session.query(models.Trajectory).filter_by(
                experiment_id=expt_id, odor_state=odor_state, clean=True).all())

            for traj in traj_array:  #np.random.permutation(traj_array):

                odors = traj.odors(session)

                if 'mosquito' in expt_id:

                    odors -= 400

                integrated_odor = odors.sum() * 0.01

                max_odor = odors.max()

                # make sure they meet our criteria

                if MIN_INTEGRATED_ODORS is not None:

                    if integrated_odor < MIN_INTEGRATED_ODORS[expt_id]:

                        continue

                if MIN_MAX_ODORS is not None:

                    if max_odor < MIN_MAX_ODORS[expt_id]:

                        continue

                trajs[odor_state].append(traj)

        # get number of pairs of trajectories from different classes

        n_traj_pairs[expt_id] = min(len(trajs['on']), len(trajs['none']))

        n_train = int(round(PROPORTION_TRAIN * n_traj_pairs[expt_id]))
        n_test = n_traj_pairs[expt_id] - n_train

        print('{} training trajs per class for expt: {}'.format(n_train, expt_id))
        print('{} test trajs per class for expt: {}'.format(n_test, expt_id))

        # loop through trials

        train_accuracies[expt_id] = {classifier[0]: [] for classifier in CLASSIFIERS}
        test_accuracies[expt_id] = {classifier[0]: [] for classifier in CLASSIFIERS}

        for tr_ctr in range(N_TRIALS):

            if (tr_ctr + 1) % 10 == 0:

                print('Trial {}'.format(tr_ctr + 1))

            # shuffled trajectory list and get subset for classifier testing

            trajs_shuffled = {
                odor_state: np.random.permutation(traj_set)[:n_traj_pairs[expt_id]]
                for odor_state, traj_set in trajs.items()
            }

            # get training and test velocity time-series

            vels_train = {
                odor_state: [traj.velocities(session) for traj in traj_set[:n_train]]
                for odor_state, traj_set in trajs_shuffled.items()
            }

            vels_test = {
                odor_state: [traj.velocities(session) for traj in traj_set[n_train:]]
                for odor_state, traj_set in trajs_shuffled.items()
            }

            for classifier in CLASSIFIERS:

                # instantiate classifier

                if classifier[0] == 'var':

                    clf = tsc.VarClassifierBinary(dim=3, order=classifier[1])

                elif classifier[0] == 'mean_speed':

                    clf = tsc.MeanSpeedClassifierBinary()

                elif classifier[0] == 'mean_heading':

                    clf = tsc.MeanHeadingClassifierBinary()

                elif classifier[0] == 'std_heading':

                    clf = tsc.StdHeadingClassifierBinary()

                # train classifier

                clf.train(positives=vels_train['on'], negatives=vels_train['none'])

                # make predictions on training set

                train_predictions = np.array(clf.predict(vels_train['on'] + vels_train['none']))
                train_correct = np.concatenate([[1] * n_train + [-1] * n_train])

                train_accuracy = 100 * np.mean(train_predictions == train_correct)

                # make predictions on test set

                # shuffle test trajectories for good luck

                rand_idxs = np.random.permutation(2 * len(vels_test['on']))

                vels_test_shuffled = np.array(vels_test['on'] + vels_test['none'])[rand_idxs]
                test_correct = np.concatenate([[1] * n_test + [-1] * n_test])[rand_idxs]

                test_predictions = np.array(
                    clf.predict(vels_test_shuffled))

                test_accuracy = 100 * np.mean(test_predictions == test_correct)

                # store results

                train_accuracies[expt_id][classifier[0]].append(train_accuracy)
                test_accuracies[expt_id][classifier[0]].append(test_accuracy)


    ## MAKE PLOTS

    fig_size = (len(EXPT_IDS) * AX_SIZE[0], len(CLASSIFIERS) * AX_SIZE[1])

    fig, axs = plt.subplots(len(CLASSIFIERS), len(EXPT_IDS), figsize=fig_size,
        sharex=True, sharey=True, tight_layout=True)

    for classifier, ax_row in zip(CLASSIFIERS, axs):

        if len(EXPT_IDS) == 1:

            ax_row = [ax_row]

        for expt_id, ax in zip(EXPT_IDS, ax_row):

            ax.hist(
                [
                    train_accuracies[expt_id][classifier[0]],
                    test_accuracies[expt_id][classifier[0]]
                ],
                bins=BINS, lw=0, color=['r', 'k'])

            if classifier == CLASSIFIERS[0]:

                ax.set_title(EXPT_LABELS[expt_id])

            if expt_id == EXPT_IDS[0]:

                ax.set_ylabel('number of trials\n({} classifier)'.format(
                    classifier[0]).replace('_', ' '))

                if classifier == CLASSIFIERS[0]:

                    ax.legend(['training', 'test'])

        if classifier == CLASSIFIERS[-1]:

            ax.set_xlabel('classification\naccuracy')

    for ax in axs.flat:

        set_fontsize(ax, FONT_SIZE)

    return fig