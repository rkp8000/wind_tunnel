from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np

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
        SEED,
        CROSSING_GROUP_IDS, CROSSING_GROUP_LABELS,
        X_0_MIN, X_0_MAX, H_0_MIN, H_0_MAX,
        T_BEFORE, T_AFTER,
        T_MODELS,
        CROSSING_GROUP_EXAMPLE_ID,
        N_DATA_POINTS_MODEL,
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

    axs[1].set_ylabel('p-value')

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

        """
        binary_model = simple_models.BinaryHeadingConcModel()

        binary_model.brute_force_fit(hs=hs, c_maxs=c_maxs, x_0s=x_0s, h_0s=h_0s)

        hs_predicted_binary = binary_model.predict(c_maxs=c_maxs, x_0s=x_0s, h_0s=h_0s)

        rss_binary = np.sum((h_0s - hs_predicted_binary) ** 2)

        binary_linear_model = simple_models.BinaryLinearHeadingConcModel()

        binary_linear_model.brute_force_fit(hs=hs, c_maxs=c_maxs, x_0s=x_0s, h_0s=h_0s)

        hs_predicted_binary_linear = binary_linear_model.predict(c_maxs=c_maxs, x_0s=x_0s, h_0s=h_0s)

        rss_binary_linear = np.sum((h_0s - hs_predicted_binary_linear) ** 2)

        f, p_val = stats.f_test(
            rss_reduced=rss_binary, rss_full=rss_binary_linear,
            df_reduced=7, df_full=8, n=n
        )

        model_infos[cg_id] = {
            'n': n,
            'rss_binary': rss_binary,
            'rss_binary_linear': rss_binary_linear,
            'f': f,
            'p_val': p_val,
        }

        print('Model fit analysis for "{}":'.format(cg_id))
        print(model_infos[cg_id])
    """

    axs.append(fig.add_subplot(2, 1, 2))

    axs[-1].scatter(
        data[CROSSING_GROUP_EXAMPLE_ID]['c_maxs'],
        data[CROSSING_GROUP_EXAMPLE_ID]['model_headings'],
        s=SCATTER_SIZE, c=SCATTER_COLOR, lw=0, alpha=SCATTER_ALPHA)

    axs[-1].set_xlabel('c_max')
    axs[-1].set_ylabel('heading at {} s\n since odor peak'.format(T_MODELS[CROSSING_GROUP_EXAMPLE_ID]))
    axs[-1].set_title('h_T vs. c_max for {}'.format(CROSSING_GROUP_LABELS[CROSSING_GROUP_EXAMPLE_ID]))

    for ax in axs:

        set_fontsize(ax, FONT_SIZE)

    return fig


def early_vs_late_heading_timecourse(
        EXPT_IDS,
        X_0_MIN, X_0_MAX, H_0_MIN, H_0_MAX,
        T_BEFORE, T_AFTER,
        FIG_SIZE, EARLY_LATE_COLORS, FONT_SIZE):
    """
    Show early vs. late headings for different experiments, along with a plot of the
    p-values for the difference between the two means.
    """

    pass


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