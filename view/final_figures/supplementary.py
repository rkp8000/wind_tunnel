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
from time_series import get_ks_p_vals

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
        figsize=(15, 10), tight_layout=True)

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

        ax.set_ylabel('time-averaged heading\ndifference (deg)')
        ax.set_title(CROSSING_GROUP_LABELS[cg.id])

    for ax in axs:
        set_fontsize(ax, FONT_SIZE)

    return fig


def early_vs_late_position_distribution(
        CROSSING_GROUP_IDS, CROSSING_GROUP_LABELS,
        X_0_MIN, X_0_MAX, H_0_MIN, H_0_MAX,
        MAX_CROSSINGS_EARLY, SUBTRACT_INITIAL_HEADING,
        T_BEFORE, T_AFTER, T_AVG_DIFF_START, T_AVG_DIFF_END,
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

                # store x_0 (uw/dw position)
                x_0s.append(crossing.feature_set_basic.position_x_peak)

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
    fig_size = (AX_SIZE[0] * AX_GRID[1], AX_SIZE[1] * AX_GRID[0])

    # position histograms
    bincs = 0.5 * (X_0_BINS[:-1] + X_0_BINS[1:])
    fig, axs = plt.subplots(*AX_GRID, figsize=fig_size, tight_layout=True)

    for cg_id, ax in zip(CROSSING_GROUP_IDS, axs.flat):

        # create early and late histograms
        handles = []

        for label, color in EARLY_LATE_COLORS.items():

            probs = np.histogram(
                x_0s_dict[cg_id][label], bins=X_0_BINS, normed=True)[0]
            handles.append(ax.plot(
                100*bincs, probs, lw=2, color=color, label=label)[0])

        ax.set_xlabel(r'$x_0$ (cm)')
        ax.set_ylabel('proportion of\ncrossings')

        title = (
            '{0},\nearly <x_0> = {1:2.3f} cm,\nlate <x_0> = {2:2.3f} cm'.format(
                CROSSING_GROUP_LABELS[cg_id],
                x_0s_dict[cg_id]['early'].mean(),
                x_0s_dict[cg_id]['late'].mean()))

        ax.set_title(title)
        ax.legend(handles=handles, loc='best')
        set_fontsize(ax, FONT_SIZE)

    return fig


def early_vs_late_heading_timecourse_given_x0_and_t_flight_scatter(
        CROSSING_GROUP_IDS, CROSSING_GROUP_LABELS,
        X_0_MIN, X_0_MAX, H_0_MIN, H_0_MAX, CROSSING_NUMBER_MAX,
        MAX_CROSSINGS_EARLY, SUBTRACT_INITIAL_HEADING,
        T_BEFORE, T_AFTER, ADJUST_NS, SCATTER_INTEGRATION_WINDOW,
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

                # calculate mean heading over integration window for scatter plot
                scatter_ys.append(np.nanmean(temp[scatter_ts[0]:scatter_ts[1]]))
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

    # history-dependence
    fig_size = (AX_SIZE[0] * AX_GRID[1], AX_SIZE[1] * AX_GRID[0])
    fig, axs = plt.subplots(*AX_GRID, figsize=fig_size, tight_layout=True)
    cc = np.concatenate
    colors = get_n_colors(CROSSING_NUMBER_MAX, colormap='jet')

    for cg_id, ax in zip(CROSSING_GROUP_IDS, axs.flat):

        # make scatter plot of x0s vs integrated headings vs crossing number
        x_0s_all = cc([x_0s_dict[cg_id]['early'], x_0s_dict[cg_id]['late']])
        t_flights_all = cc(
            [t_flights_dict[cg_id]['early'], t_flights_dict[cg_id]['late']])
        ys_all = cc([scatter_ys_dict[cg_id]['early'], scatter_ys_dict[cg_id]['late']])
        cs_all = cc([crossing_ns_dict[cg_id]['early'], crossing_ns_dict[cg_id]['late']])

        cs = np.array([colors[c-1] for c in cs_all])

        hs = []

        for c in sorted(np.unique(cs_all)):
            label = 'cn = {}'.format(c)
            mask = cs_all == c
            h = ax.scatter(100*x_0s_all[mask], ys_all[mask],
                s=20, c=cs[mask], lw=0, label=label)
            hs.append(h)

        # calculate partial correlation between crossing number and heading given x
        not_nan = ~np.isnan(ys_all)
        r, p = stats.partial_corr(
            cs_all[not_nan], ys_all[not_nan],
            controls=[x_0s_all[not_nan], t_flights_all[not_nan]])

        ax.set_xlabel('x (cm)')
        ax.set_ylabel('$<\Delta$h> (deg)'.format(
            *SCATTER_INTEGRATION_WINDOW))

        title = CROSSING_GROUP_LABELS[cg_id] + \
            ', R = {0:.2f}, P = {1:.3f}'.format(r, p)
        ax.set_title(title)

        ax.legend(handles=hs, loc='upper center', ncol=3)
        set_fontsize(ax, 16)

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


def per_trajectory_early_late_diff_analysis(
        CROSSING_GROUP_IDS, CROSSING_GROUP_LABELS,
        X_0_MIN, X_0_MAX, H_0_MIN, H_0_MAX, CROSSING_NUMBER_MAX,
        MAX_CROSSINGS_EARLY, SUBTRACT_INITIAL_HEADING,
        T_BEFORE, T_AFTER, ADJUST_NS, SCATTER_INTEGRATION_WINDOW,
        AX_SIZE, AX_GRID, EARLY_LATE_COLORS, ALPHA,
        P_VAL_COLOR, P_VAL_Y_LIM, LEGEND_CROSSING_GROUP_ID,
        FONT_SIZE):
    """
    Calculate the early vs. late difference on a per-trajectory basis,
    i.e., directly comparing late crossings in a trajectory to early
    crossings in the same trajectory.
    """
    ts_before = int(round(T_BEFORE / DT))
    ts_after = int(round(T_AFTER / DT))

    fig, axs_ = plt.subplots(
        *AX_GRID, figsize=(15, 10), tight_layout=True, squeeze=False)
    axs = {cg_id: ax for cg_id, ax in zip(CROSSING_GROUP_IDS, axs_.flatten())}

    for cg_id in CROSSING_GROUP_IDS:

        print('LOADING DATA FOR CROSSING GROUP: "{}"...'.format(cg_id))

        # get all crossing objects
        crossings_all = session.query(models.Crossing).join(
            models.CrossingFeatureSetBasic).filter(
            models.Crossing.crossing_group_id == cg_id,
            models.Crossing.crossing_number <= CROSSING_NUMBER_MAX,
            models.CrossingFeatureSetBasic.position_x_peak.between(
                X_0_MIN, X_0_MAX),
            models.CrossingFeatureSetBasic.heading_xyz_peak.between(
                H_0_MIN, H_0_MAX))

        headings = []
        x_0s = []
        t_flights = []

        crossing_types = []
        traj_ids = []

        # loop through crossings, appropriately adding them to data structure
        for crossing in crossings_all:
            # skip if exclusion criteria met
            x_0 = crossing.feature_set_basic.position_x_peak
            h_0 = crossing.feature_set_basic.heading_xyz_peak

            # figure out whether crossing is early or late
            if crossing.crossing_number <= MAX_CROSSINGS_EARLY:
                crossing_type = 'early'
            else:
                crossing_type = 'late'

            # get headings
            headings_ = crossing.timepoint_field(
                session, 'heading_xyz', -ts_before, ts_after - 1,
                'peak', 'peak', nan_pad=True)

            if SUBTRACT_INITIAL_HEADING:
                headings_ -= h_0

            # store headings, x, and t_flight in large crossings list
            headings.append(deepcopy(headings_))
            x_0s.append(x_0)
            t_flights.append(crossing.t_flight_peak)

            crossing_types.append(crossing_type)
            traj_ids.append(crossing.trajectory_id)

        headings = np.array(headings)
        x_0s = np.array(x_0s)
        t_flights = np.array(t_flights)

        print('CALCULATING HEADING RESIDUALS...')
        # fit linear model for each time point
        # and calculate residual headings
        h_res = np.nan * np.zeros(headings.shape)

        for t_step in range(ts_before + ts_after):

            # get targets and predictors
            targs = headings[:, t_step]
            predictors = np.array([x_0s, t_flights]).T
            not_nan = ~np.isnan(targs)

            # fit model
            lm = linear_model.LinearRegression(n_jobs=-1)
            lm.fit(predictors[not_nan], targs[not_nan])

            predictions = np.nan * np.zeros(len(targs))
            predictions[not_nan] = lm.predict(predictors[not_nan])

            h_res_ = targs - predictions
            h_res[:, t_step] = h_res_

        print('PRUNING UNPAIRED CROSSINGS...')
        # create new dict with traj_ids as keys and sub-dicts as values
        # containing lists of early vs. late crossings
        crossings_dict = {}

        for h_res_, crossing_type, traj_id in zip(
                h_res, crossing_types, traj_ids):

            # make new item in dict if trajectory id not already in it
            if traj_id not in crossings_dict:
                crossings_dict[traj_id] = {'early': [], 'late': []}

            # store crossing heading residuals under correct traj and
            # crossing type
            crossings_dict[traj_id][crossing_type].append(h_res_)

        # get pairs of pre-averaged crossing types
        # in trajectories that include both valid early and late
        # crossings
        earlies = []
        lates = []
        for traj_id, crossing_sets in crossings_dict.items():

            if crossing_sets['early'] and crossing_sets['late']:
                # average over multiple early/multiple late encounters
                early = np.nanmean(crossing_sets['early'], axis=0)
                late = np.nanmean(crossing_sets['late'], axis=0)

                earlies.append(early.copy())
                lates.append(late.copy())

        # calculate late-minus-early difference
        earlies = np.array(earlies)
        lates = np.array(lates)
        diffs = lates - earlies

        print('CROSSING GROUP: "{}"'.format(cg_id))
        print('{} TRAJECTORIES INCLUDING VALID EARLY AND LATE CROSSINGS'.
            format(len(diffs)))

        # calculate p-values using paired t-test
        p_vals = np.nan * np.zeros(diffs.shape[1])
        for t_step in np.arange(ts_before + ts_after):
            a = earlies[:, t_step]
            b = lates[:, t_step]

            not_nan_a = ~np.isnan(a)
            not_nan_b = ~np.isnan(b)
            not_nan = (not_nan_a * not_nan_b).astype(bool)

            p_val = ttest_rel(a[not_nan], b[not_nan])[1]
            p_vals[t_step] = p_val

        # plot earlies and lates, overlaid with p-values
        ts = np.arange(-ts_before, ts_after) / 100.
        ax = axs[cg_id]

        # early
        h_1 = ax.plot(
            ts, np.nanmean(earlies, axis=0), color='b', lw=2, label='early')[0]
        ax.fill_between(
            ts,
            np.nanmean(earlies, axis=0) - stats.nansem(earlies, axis=0),
            np.nanmean(earlies, axis=0) + stats.nansem(earlies, axis=0),
            color='b', alpha=0.2)

        # late
        h_2 = ax.plot(
            ts, np.nanmean(lates, axis=0), color='g', lw=2, label='late')[0]
        ax.fill_between(
            ts,
            np.nanmean(lates, axis=0) - stats.nansem(lates, axis=0),
            np.nanmean(lates, axis=0) + stats.nansem(lates, axis=0),
            color='g', alpha=0.2)

        # p-values
        ax_twin = ax.twinx()
        ax_twin.plot(ts, p_vals, lw=2, ls='--', color='k')
        ax_twin.axhline(0.05, color='gray')

        ax.set_xlabel('time since crossing (s)')
        ax.set_ylabel('h*')
        ax.set_title(CROSSING_GROUP_LABELS[cg_id])
        ax.legend(handles=[h_1, h_2], loc='best')

        ax.set_xlim(ts[0], ts[-1])
        set_fontsize(ax, 16)

    return fig


def early_crossings_vs_n_crossings(
        CROSSING_GROUP_IDS, CROSSING_GROUP_LABELS,
        X_0_MIN, X_0_MAX, H_0_MIN, H_0_MAX,
        MAX_CROSSINGS_EARLY, SUBTRACT_INITIAL_HEADING,
        T_BEFORE, T_AFTER, AX_GRID):
    """
    Plot early crossings only as a function of how many crossings there
    are in a trajectory in total. This tests the null hypothesis that there
    are two types of flies: ones that perform only a few upwind oriented
    crossings and ones that perform many crosswind oriented crossings.
    """
    ts_before = int(round(T_BEFORE / DT))
    ts_after = int(round(T_AFTER / DT))

    fig, axs_ = plt.subplots(
        *AX_GRID, figsize=(15, 10), tight_layout=True, squeeze=False)
    axs = {cg_id: ax for cg_id, ax in zip(CROSSING_GROUP_IDS, axs_.flatten())}

    for cg_id in CROSSING_GROUP_IDS:

        print('LOADING DATA FOR CROSSING GROUP: "{}"...'.format(cg_id))

        # get all early crossings that satisfy our criteria
        crossings_early = session.query(models.Crossing).join(
            models.CrossingFeatureSetBasic).filter(
            models.Crossing.crossing_group_id == cg_id,
            models.Crossing.crossing_number <= MAX_CROSSINGS_EARLY,
            models.CrossingFeatureSetBasic.position_x_peak.between(
                X_0_MIN, X_0_MAX),
            models.CrossingFeatureSetBasic.heading_xyz_peak.between(
                H_0_MIN, H_0_MAX))

        # store headings, x_0, t_flight, and total number of crossings in
        # trajectory for each early crossing
        headings = np.nan * np.zeros(
            (crossings_early.count(), ts_before + ts_after))
        x_0s = np.nan * np.zeros(crossings_early.count())
        t_flights = np.nan * np.zeros(crossings_early.count())
        n_crossings_traj = -1 * np.zeros(crossings_early.count(), dtype=int)
        traj_ids = []

        for ctr, crossing in enumerate(crossings_early):
            # get headings
            headings_ = crossing.timepoint_field(
                session, 'heading_xyz', -ts_before, ts_after-1,
                'peak', 'peak', nan_pad=True)

            if SUBTRACT_INITIAL_HEADING:
                headings_ -= crossing.feature_set_basic.heading_xyz_peak

            # get total number of crossings in the trajectory this
            # crossing came from
            traj_id = crossing.trajectory_id

            n_crossings_traj_ = session.query(models.Crossing).filter(
                models.Crossing.crossing_group_id == cg_id,
                models.Crossing.trajectory_id == traj_id).count()

            # store headings, x pos, t_flight, and n crossings in traj
            headings[ctr] = deepcopy(headings_)
            x_0s[ctr] = crossing.feature_set_basic.position_x_peak
            t_flights[ctr] = crossing.t_flight_peak
            n_crossings_traj[ctr] = n_crossings_traj_

            traj_ids.append(traj_id)

        traj_ids = np.array(traj_ids)

        print('CALCULATING HEADING RESIDUALS...')
        h_res = np.nan * np.zeros(headings.shape)

        for t_step in range(ts_before + ts_after):

            # get targets (headings) and predictors (x_0 and t_flight)
            targs = headings[:, t_step]
            predictors = np.array([x_0s, t_flights]).T
            not_nan = ~np.isnan(targs)

            # fit model
            lm = linear_model.LinearRegression(n_jobs=-1)
            lm.fit(predictors[not_nan], targs[not_nan])

            predictions = np.nan * np.zeros(len(targs))
            predictions[not_nan] = lm.predict(predictors[not_nan])

            h_res_ = targs - predictions
            h_res[:, t_step] = h_res_

        # sort heading time-series into groups according to total crossings
        # in trajectory
        traj_short_mask = n_crossings_traj <= MAX_CROSSINGS_EARLY
        traj_long_mask = n_crossings_traj > MAX_CROSSINGS_EARLY

        h_res_traj_short = h_res[traj_short_mask]
        h_res_traj_long = h_res[traj_long_mask]

        print('{} EARLY CROSSINGS IN SHORT TRAJECTORY GROUP'.format(
            len(h_res_traj_short)))
        print('{} EARLY CROSSINGS IN LONG TRAJECTORY GROUP'.format(
            len(h_res_traj_long)))

        # get p-values using adjusted t-test
        p_vals = np.nan * np.zeros(h_res_traj_short.shape[1])
        for t_step in np.arange(ts_before + ts_after):
            a = h_res_traj_short[:, t_step]
            b = h_res_traj_long[:, t_step]

            not_nan_a = ~np.isnan(a)
            not_nan_b = ~np.isnan(b)

            n1 = len(np.unique(traj_ids[traj_short_mask][not_nan_a]))
            n2 = len(np.unique(traj_ids[traj_long_mask][not_nan_b]))

            p_val = stats.ttest_adjusted_ns(
                a[not_nan_a], b[not_nan_b], n1, n2)[1]
            p_vals[t_step] = p_val

        # make plots
        ts = np.arange(-ts_before, ts_after) / 100.
        ax = axs[cg_id]
        ax_twin = ax.twinx()

        # trajs with few crossings
        h_1 = ax.plot(
            ts, np.nanmean(h_res_traj_short, axis=0),
            color='r', lw=2, label='few')[0]
        ax.fill_between(
            ts,
            np.nanmean(h_res_traj_short, axis=0)
                - stats.nansem(h_res_traj_short, axis=0),
            np.nanmean(h_res_traj_short, axis=0)
                + stats.nansem(h_res_traj_short, axis=0),
            color='r', alpha=0.2)

        # trajs with many crossings
        h_2 = ax.plot(
            ts, np.nanmean(h_res_traj_long, axis=0),
            color='c', lw=2, label='many')[0]
        ax.fill_between(
            ts,
            np.nanmean(h_res_traj_long, axis=0)
                - stats.nansem(h_res_traj_long, axis=0),
            np.nanmean(h_res_traj_long, axis=0)
                + stats.nansem(h_res_traj_long, axis=0),
            color='c', alpha=0.2)

        ax.set_xlabel('time since crossing (s)')
        ax.set_ylabel('h*')
        ax.set_title(CROSSING_GROUP_LABELS[cg_id])
        ax.legend(handles=[h_1, h_2])

        ax.set_xlim(ts[0], ts[-1])

        # plot p-values
        ax_twin.plot(ts, p_vals, color='k', lw=2, ls='--')
        ax_twin.axhline(0.05, color='gray', lw=2)
        ax_twin.set_ylabel('p-value (t-test)')

        for ax_ in [ax, ax_twin]:
            set_fontsize(ax_, 16)

    return fig


def models_vs_data_mean_history_dependence(
        DATA_TEMP_FILE,
        SURGE_CAST_TEMP_FILE,
        CENTERLINE_TEMP_FILE,
        INFOTAXIS_TEMP_FILE,
        INFOTAXIS_DT,
        T_MIN, T_MAX):
    """
    Plot the difference between the model and data mean crossing-triggered
    heading time-series for early and late crossings.
    :param DATA_TEMP_FILE:
    :param SURGE_CAST_TEMP_FILE:
    :param CENTERLINE_TEMP_FILE:
    :param INFOTAXIS_TEMP_FILE:
    :param INFOTAXIS_DT:
    :return:
    """
    data = np.load(DATA_TEMP_FILE)[0]
    surge_cast = np.load(SURGE_CAST_TEMP_FILE)[0]
    centerline = np.load(CENTERLINE_TEMP_FILE)[0]
    infotaxis = np.load(INFOTAXIS_TEMP_FILE)[0]

    fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharey=True, tight_layout=True)

    ts = np.arange(T_MIN, T_MAX, 0.01)
    for means in [data, surge_cast, centerline]:
        assert np.max(means['t']) > T_MAX
        assert np.min(means['t']) < T_MIN

    assert np.max(infotaxis['t'] * INFOTAXIS_DT) > T_MAX
    assert np.min(infotaxis['t'] * INFOTAXIS_DT) < T_MIN

    for label, ax in zip(['early', 'late'], axs):
        # resample all data
        mask_data = ((data['t'] >= ts.min()) \
            * (data['t'] < ts.max())).astype(bool)
        t_data = data['t'][mask_data]
        mean_data = data[label][mask_data]
        mean_data, t_data = resample(mean_data, len(ts), t=t_data)

        mask_surge_cast = ((surge_cast['t'] >= ts.min()) \
            * (surge_cast['t'] < ts.max())).astype(bool)
        t_surge_cast = surge_cast['t'][mask_surge_cast]
        mean_surge_cast = surge_cast[label][mask_surge_cast]
        mean_surge_cast, t_surge_cast = resample(
            mean_surge_cast, len(ts), t=t_surge_cast)

        mask_centerline = ((centerline['t'] >= ts.min()) \
            * (centerline['t'] < ts.max())).astype(bool)
        t_centerline = centerline['t'][mask_centerline]
        mean_centerline = centerline[label][mask_centerline]
        mean_centerline, t_centerline = resample(
            mean_centerline, len(ts), t=t_centerline)

        t_infotaxis = infotaxis['t'] * INFOTAXIS_DT
        mask_infotaxis = ((t_infotaxis >= ts.min()) \
            * (t_infotaxis < ts.max())).astype(bool)
        t_infotaxis = t_infotaxis[mask_infotaxis]
        mean_infotaxis = infotaxis[label][mask_infotaxis]
        mean_infotaxis, t_infotaxis = resample(
            mean_infotaxis, len(ts), t=t_infotaxis)

        # plot difference between data and all models
        hs = []
        hs.append(
            ax.plot(t_data, mean_surge_cast - mean_data,
                lw=2, label='surge/cast')[0]
        )

        hs.append(
            ax.plot(t_data, mean_centerline - mean_data,
                lw=2, label='centerline')[0]
        )

        hs.append(
            ax.plot(t_data, mean_infotaxis - mean_data,
                lw=2, label='infotaxis')[0]
        )

        ax.axhline(0, color='gray', ls='--')
        ax.set_title('{} crossings'.format(label))
        ax.set_xlabel('time since crossing (s)')
        ax.set_ylabel(r'model $\Delta$ heading - data $\Delta$ heading')
        ax.legend(handles=hs)

    for ax in axs:
        set_fontsize(ax, 16)

    return fig

