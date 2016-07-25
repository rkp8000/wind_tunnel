from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
from scipy.stats import ks_2samp

from db_api import models
from db_api.connect import session

from axis_tools import set_fontsize
import simple_models
import stats

from experimental_constants import DT, PLUME_PARAMS_DICT


def trajectory_example_and_visualization_of_crossing_variability(
        SEED,
        EXPT_ID,
        TRAJ_NUMBER, VISUAL_THRESHOLD,
        CROSSING_NUMBERS, T_BEFORE, T_AFTER,
        FIG_SIZE, FONT_SIZE):
    """
    Show an example trajectory through a wind tunnel plume with the crossings marked.
    Show many crossings overlaid on the plume in 3D and show the mean peak-triggered heading
    with its SEM as well as many individual examples.
    """

    pass


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

            c_maxs.append(crossing.max_odor)
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
        axs[0].fill_between(t, data[cg_id]['lbs'], data[cg_id]['ubs'], color=color, alpha=0.2)

        handles.append(handle)

        # show p-values

        axs[1].plot(t[t > 0], data[cg_id]['p_vals'][t > 0], color=color, lw=2, ls='--')

    axs[0].set_xlim(-T_BEFORE, T_AFTER)

    axs[0].set_xlabel('time of heading measurement\nsince odor peak (s)')
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

    axs[-1].set_xlabel('c_max')
    axs[-1].set_ylabel('heading at {} s\n since odor peak'.format(T_MODELS[CROSSING_GROUP_EXAMPLE_ID]))
    axs[-1].set_title('h_T vs. c_max for {}'.format(CROSSING_GROUP_LABELS[CROSSING_GROUP_EXAMPLE_ID]))

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

        ax.set_xlabel('time since odor peak (s)')

        if SUBTRACT_INITIAL_HEADING:

            ax.set_ylabel('$\Delta$ heading (degrees)')

        else:

            ax.set_ylabel('heading (degrees)')

        ax.set_title(CROSSING_GROUP_LABELS[cg_id])

        if cg_id == LEGEND_CROSSING_GROUP_ID:

            ax.legend(handles=handles, loc='lower left')

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
        WIND_SPEED_SIM_IDS,
        HISTORY_DEPENDENCE_SIM_IDS,
        HEAT_MAP_EXPT_IDS, HEAT_MAP_SIM_IDS,
        X_0_MIN, X_0_MAX, H_0_MIN, H_0_MAX,
        T_BEFORE_EXPT, T_AFTER_EXPT,
        T_BEFORE_SIM, T_AFTER_SIM,
        FIG_SIZE, FONT_SIZE,
        EXPT_COLORS, SIMULATION_COLORS):
    """
    Show infotaxis-generated trajectories alongside empirical trajectories. Show wind-speed
    dependence and history dependence.
    """

    pass


def classifier(
        SEED,
        EXPT_IDS,
        CLASSIFIER_TYPES,
        N_TRAINING, N_TEST, N_TRIALS,
        INTEGRATED_ODOR_THRESHOLDS,
        AX_SIZE, AX_GRID, FONT_SIZE):
    """
    Show classification accuracy when classifying whether insect is engaged in odor-tracking
    or not.
    """

    pass