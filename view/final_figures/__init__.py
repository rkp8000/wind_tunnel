from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np

from experimental_constants import PLUME_PARAMS_DICT


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
        EXPT_IDS,
        X_0_MIN, X_0_MAX, H_0_MIN, H_0_MAX,
        T_BEFORE, T_AFTER,
        T_MODEL,
        N_DATA_POINTS_MODEL,
        FIG_SIZE, EXPT_COLORS,
        SCATTER_SIZE, SCATTER_COLOR, SCATTER_ALPHA,
        FONT_SIZE):
    """
    Show a partial correlation plot between concentration and heading a little while after the
    peak odor concentration. Show the relationship between peak concentration and heading
    at a specific time (T_MODEL) post peak via a scatter plot.

    Then fit a binary threshold model and a model with a linear piece to the data and see if
    the linear one fits significantly better.
    """

    pass


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