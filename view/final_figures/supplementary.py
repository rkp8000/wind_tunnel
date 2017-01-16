from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn import linear_model
from db_api.connect import session
from db_api import models
from scipy.stats import ks_2samp

import stats
from axis_tools import set_fontsize
import simple_models
from plot import get_n_colors

from experimental_constants import DT


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


def early_vs_late_heading_timecourse_x0_accounted_for(
        CROSSING_GROUP_IDS, CROSSING_GROUP_LABELS,
        X_0_MIN, X_0_MAX, H_0_MIN, H_0_MAX, CROSSING_NUMBER_MAX,
        MAX_CROSSINGS_EARLY, SUBTRACT_INITIAL_HEADING,
        T_BEFORE, T_AFTER, SCATTER_INTEGRATION_WINDOW,
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
    scatter_ts = [ts_before + int(round(t / DT)) for t in SCATTER_INTEGRATION_WINDOW]

    # loop over crossing groups
    x_0s_dict = {}
    headings_dict = {}
    residuals_dict = {}
    p_vals_dict = {}

    scatter_ys_dict = {}
    crossing_ns_dict = {}

    for cg_id in CROSSING_GROUP_IDS:

        # get crossing group
        crossing_group = session.query(models.CrossingGroup).filter_by(id=cg_id).first()

        # get early and late crossings
        crossings_dict = {}
        crossings_all = session.query(models.Crossing).filter_by(
            crossing_group=crossing_group)
        crossings_dict['early'] = crossings_all.filter(
            models.Crossing.crossing_number <= MAX_CROSSINGS_EARLY)
        crossings_dict['late'] = crossings_all.filter(
            models.Crossing.crossing_number > MAX_CROSSINGS_EARLY,
            models.Crossing.crossing_number <= CROSSING_NUMBER_MAX)

        x_0s_dict[cg_id] = {}
        headings_dict[cg_id] = {}

        scatter_ys_dict[cg_id] = {}
        crossing_ns_dict[cg_id] = {}

        for label in ['early', 'late']:

            x_0s = []
            headings = []
            scatter_ys = []
            crossing_ns = []

            # get all initial headings, initial xs, peak concentrations, and heading time-series
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
                h_0 = getattr(
                    crossing.feature_set_basic, 'heading_xyz_{}'.format('peak'))

                if not (X_0_MIN <= x_0 <= X_0_MAX): continue
                if not (H_0_MIN <= h_0 <= H_0_MAX): continue

                # store x_0 (uw/dw position)
                x_0s.append(x_0)

                # get and store headings
                temp = crossing.timepoint_field(
                    session, 'heading_xyz', -ts_before, ts_after - 1,
                    'peak', 'peak', nan_pad=True)

                # subtract initial heading if desired
                if SUBTRACT_INITIAL_HEADING: temp -= temp[ts_before]

                # store headings
                headings.append(temp)

                # calculate mean heading over integration window for scatter plot
                scatter_ys.append(np.nanmean(temp[scatter_ts[0]:scatter_ts[1]]))
                crossing_ns.append(crossing.crossing_number)

            x_0s_dict[cg_id][label] = np.array(x_0s).copy()
            headings_dict[cg_id][label] = np.array(headings).copy()

            scatter_ys_dict[cg_id][label] = np.array(scatter_ys).copy()
            crossing_ns_dict[cg_id][label] = np.array(crossing_ns).copy()

        x_early = x_0s_dict[cg_id]['early']
        x_late = x_0s_dict[cg_id]['late']
        h_early = headings_dict[cg_id]['early']
        h_late = headings_dict[cg_id]['late']

        x0s_all = np.concatenate([x_early, x_late])
        hs_all = np.concatenate([h_early, h_late], axis=0)

        residuals_dict[cg_id] = {
            'early': np.nan * np.zeros(h_early.shape),
            'late': np.nan * np.zeros(h_late.shape),
        }

        # fit heading linear prediction from x0 at each time point
        # and subtract from original heading
        for t_step in range(ts_before + ts_after):

            # get all headings for this time point
            hs_t = hs_all[:, t_step]
            residuals = np.nan * np.zeros(hs_t.shape)

            # only use headings that exist
            not_nan = ~np.isnan(hs_t)

            # fit linear model
            rgr = linear_model.LinearRegression()
            rgr.fit(x0s_all[not_nan][:, None], hs_t[not_nan])

            residuals[not_nan] = hs_t[not_nan] - rgr.predict(x0s_all[not_nan][:, None])

            assert np.all(np.isnan(residuals) == np.isnan(hs_t))

            r_early, r_late = np.split(residuals, [len(x_early)])
            residuals_dict[cg_id]['early'][:, t_step] = r_early
            residuals_dict[cg_id]['late'][:, t_step] = r_late

        # loop through all time points and calculate p-value (ks-test)
        # between early and late
        p_vals = []

        for t_step in range(ts_before + ts_after):

            early_with_nans = residuals_dict[cg_id]['early'][:, t_step]
            late_with_nans = residuals_dict[cg_id]['late'][:, t_step]

            early_no_nans = early_with_nans[~np.isnan(early_with_nans)]
            late_no_nans = late_with_nans[~np.isnan(late_with_nans)]

            # calculate statistical significance
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
            headings_mean = np.nanmean(residuals_dict[cg_id][label], axis=0)
            headings_sem = stats.nansem(residuals_dict[cg_id][label], axis=0)

            handles.append(ax.plot(
                t, headings_mean, color=color, lw=2, label=label, zorder=1)[0])
            ax.fill_between(
                t, headings_mean - headings_sem, headings_mean + headings_sem,
                color=color, alpha=ALPHA, zorder=1)

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
        ax_twin.set_ylabel('p-value (KS test)', fontsize=FONT_SIZE)

        set_fontsize(ax_twin, FONT_SIZE)

    fig_1, axs_1 = plt.subplots(*AX_GRID, figsize=fig_size, tight_layout=True)
    cc = np.concatenate
    colors = get_n_colors(CROSSING_NUMBER_MAX, colormap='jet')

    for cg_id, ax in zip(CROSSING_GROUP_IDS, axs_1.flat):

        # make scatter plot of x0s vs integrated headings vs crossing number
        x_0s_all = cc([x_0s_dict[cg_id]['early'], x_0s_dict[cg_id]['late']])
        ys_all = cc([scatter_ys_dict[cg_id]['early'], scatter_ys_dict[cg_id]['late']])
        cs_all = cc([crossing_ns_dict[cg_id]['early'], crossing_ns_dict[cg_id]['late']])

        cs = np.array([colors[c-1] for c in cs_all])

        hs = []

        for c in sorted(np.unique(cs_all)):
            label = 'cn = {}'.format(c)
            mask = cs_all == c
            h = ax.scatter(x_0s_all[mask], ys_all[mask],
                s=20, c=cs[mask], lw=0, label=label)
            hs.append(h)

        # calculate partial correlation between crossing number and heading given x
        not_nan = ~np.isnan(ys_all)
        r, p = stats.partial_corr(
            cs_all[not_nan], ys_all[not_nan], controls=[x_0s_all[not_nan]])

        ax.set_xlabel('x')
        ax.set_ylabel(r'$\Delta$h_mean({}:{}) (deg)'.format(
            *SCATTER_INTEGRATION_WINDOW))

        title = CROSSING_GROUP_LABELS[cg_id] + \
            ', R = {0:.2f}, P = {1:.3f}'.format(r, p)
        ax.set_title(title)

        ax.legend(handles=hs, loc='upper center', ncol=3)
        set_fontsize(ax, 16)

    return fig_0


def heading_duration_dependence(
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
        ds = []
        headings = []

        for crossing in crossings_all:

            # throw away crossings that do not meet trigger criteria

            position_x = getattr(
                crossing.feature_set_basic, 'position_x_{}'.format('exit'))

            if not (X_0_MIN <= position_x <= X_0_MAX): continue

            heading_xyz = getattr(
                crossing.feature_set_basic, 'heading_xyz_{}'.format('exit'))

            if not (H_0_MIN <= heading_xyz <= H_0_MAX): continue

            ds.append(crossing.duration)
            x_0s.append(position_x)
            h_0s.append(heading_xyz)

            temp = crossing.timepoint_field(
                session, 'heading_xyz', -ts_before, ts_after - 1,
                'exit', 'exit', nan_pad=True)

            headings.append(temp)

        x_0s = np.array(x_0s)
        h_0s = np.array(h_0s)
        ds = np.array(ds)
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
                ds[mask], headings_this_tp[mask],
                [x_0s[mask], h_0s[mask]])

            partial_corrs[ts] = r
            p_vals[ts] = p
            lbs[ts] = lb
            ubs[ts] = ub

        data[cg_id] = {
            'x_0s': x_0s,
            'h_0s': h_0s,
            'ds': ds,
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
    axs[0].set_ylabel('heading-duration\npartial correlation')
    axs[0].legend(handles=handles, loc='upper left')

    axs[1].set_ylim(0, 0.2)

    axs[1].set_ylabel('p-value (dashed lines)')

    ## FIT BOTH MODELS TO EACH DATASET

    model_infos = {cg_id: None for cg_id in CROSSING_GROUP_IDS}

    for cg_id in CROSSING_GROUP_IDS:

        hs = data[cg_id]['model_headings']
        ds = data[cg_id]['ds']
        x_0s = data[cg_id]['x_0s']
        h_0s = data[cg_id]['h_0s']

        valid_mask = ~np.isnan(hs)

        hs = hs[valid_mask]
        ds = ds[valid_mask]
        x_0s = x_0s[valid_mask]
        h_0s = h_0s[valid_mask]

        n = len(hs)

        rho = stats.pearsonr_partial_with_confidence(ds, hs, [x_0s, h_0s])[0]

        binary_model = simple_models.ThresholdLinearHeadingConcModel(
            include_c_max_coefficient=False)

        binary_model.brute_force_fit(hs=hs, c_maxs=ds, x_0s=x_0s, h_0s=h_0s)

        hs_predicted_binary = binary_model.predict(c_maxs=ds, x_0s=x_0s, h_0s=h_0s)

        rss_binary = np.sum((hs - hs_predicted_binary) ** 2)

        threshold_linear_model = simple_models.ThresholdLinearHeadingConcModel(
            include_c_max_coefficient=True)

        threshold_linear_model.brute_force_fit(hs=hs, c_maxs=ds, x_0s=x_0s, h_0s=h_0s)

        hs_predicted_threshold_linear = threshold_linear_model.predict(
            c_maxs=ds, x_0s=x_0s, h_0s=h_0s)

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
            'h_vs_d_coef': threshold_linear_model.linear_models['above'].coef_[0],
            'rho': rho,
        }

        pprint('Model fit analysis for "{}":'.format(cg_id))
        pprint(model_infos[cg_id])

    axs.append(fig.add_subplot(2, 1, 2))

    axs[-1].scatter(
        data[CROSSING_GROUP_EXAMPLE_ID]['ds'],
        data[CROSSING_GROUP_EXAMPLE_ID]['model_headings'],
        s=SCATTER_SIZE, c=SCATTER_COLOR, lw=0, alpha=SCATTER_ALPHA)

    axs[-1].set_xlim(0, 0.5)
    axs[-1].set_ylim(0, 180)

    axs[-1].set_xlabel('crossing duration (s)')
    axs[-1].set_ylabel('heading at {} s\n since crossing'.format(
        T_MODELS[CROSSING_GROUP_EXAMPLE_ID]))
    axs[-1].set_title('heading-duration relationship for {}'.format(
        CROSSING_GROUP_LABELS[CROSSING_GROUP_EXAMPLE_ID]))

    for ax in axs:

        set_fontsize(ax, FONT_SIZE)

    return fig

