from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle

import axis_tools

from db_api import models
from db_api.connect import session

import time_series_classifier as tsc
import insect_glm_fit_helper as igfh


EXPERIMENT_ID = 'fruitfly_0.4mps_checkerboard_floor'
EXPERIMENT_ID = 'mosquito_0.4mps_checkerboard_floor'

ODOR_STATES = {'on', 'none'}

INTEGRATED_ODOR_THRESHOLDS = {
    'on': 30,
    'none': 30,
}

INTEGRATED_ODOR_THRESHOLDS = {
    'on': 30,
    'none': 30,
}

N_TRAIN = 8  # 50
N_TEST = 4  # 100
N_TRIALS = 20  #200

FACE_COLOR = 'white'
FIG_SIZE = (10, 12)
FONT_SIZE = 20

METHODS = ['var', 'mean_speed', 'mean_heading', 'std_heading']


def main():

    train_prediction_accuracy = {}
    test_prediction_accuracy = {}

    for condition in ['experimental', 'control']:

        print('Condition: {}'.format(condition))

        train_prediction_accuracy[condition] = {method: [] for method in METHODS}
        test_prediction_accuracy[condition] = {method: [] for method in METHODS}

        trajs = {}
        if condition == 'control':
            trajs['on'] = igfh.get_trajs_with_integrated_odor_above_threshold(
                EXPERIMENT_ID, 'on',
                integrated_odor_threshold=INTEGRATED_ODOR_THRESHOLDS['on'],
                max_trajs=N_TRAIN+N_TEST,
            )
            trajs['none'] = igfh.get_trajs_with_integrated_odor_above_threshold(
                EXPERIMENT_ID, 'on',
                integrated_odor_threshold=INTEGRATED_ODOR_THRESHOLDS['none'],
                max_trajs=N_TRAIN+N_TEST,
            )

        elif condition == 'experimental':
            trajs['on'] = igfh.get_trajs_with_integrated_odor_above_threshold(
                EXPERIMENT_ID, 'on',
                integrated_odor_threshold=INTEGRATED_ODOR_THRESHOLDS['on'],
                max_trajs=N_TRAIN+N_TEST,
            )
            trajs['none'] = igfh.get_trajs_with_integrated_odor_above_threshold(
                EXPERIMENT_ID, 'none',
                integrated_odor_threshold=INTEGRATED_ODOR_THRESHOLDS['none'],
                max_trajs=N_TRAIN+N_TEST,
            )

        print('{} trajectories with odor on'.format(len(trajs['on'])))
        print('{} trajectories with odor off'.format(len(trajs['none'])))

        assert len(trajs['on']) >= N_TRAIN + N_TEST
        assert len(trajs['none']) >= N_TRAIN + N_TEST

        print('Sufficient trajectories for classification analysis')


        for tr_ctr in range(N_TRIALS):

            if tr_ctr % 20 == 19:
                print('Trial # {}'.format(tr_ctr + 1))

            vels = {}

            # get all data
            for odor_state in ODOR_STATES:

                shuffle(trajs[odor_state])

                vels[odor_state] = {
                    'train': [traj.velocities(session) for traj in trajs[odor_state][:N_TRAIN]],
                    'test': [traj.velocities(session) for traj in trajs[odor_state][N_TRAIN:N_TRAIN+N_TEST]],
                }

            # loop through all classifiers
            for method in METHODS:
                # train classifer
                if method == 'var':
                    clf = tsc.VarClassifierBinary(dim=3, order=2)
                elif method == 'mean_speed':
                    clf = tsc.MeanSpeedClassifierBinary()
                elif method == 'mean_heading':
                    clf = tsc.MeanHeadingClassifierBinary()
                elif method == 'std_heading':
                    clf = tsc.StdHeadingClassifierBinary()

                clf.train(positives=vels['on']['train'], negatives=vels['none']['train'])
                # make predictions on training set
                train_predictions = np.array(clf.predict(vels['on']['train'] + vels['none']['train']))
                train_ground_truth = np.concatenate([[1] * N_TRAIN, [-1] * N_TRAIN])

                train_accuracy = 100 * np.mean(train_predictions == train_ground_truth)

                # make predictions on test set
                test_trajs = np.array(vels['on']['test'] + vels['none']['test'])
                test_ground_truth = np.concatenate([[1] * N_TEST, [-1] * N_TEST])

                # shuffle trajectories and ground truths for good luck
                rand_idx = np.random.permutation(len(test_trajs))
                test_trajs = test_trajs[rand_idx]
                test_ground_truth = test_ground_truth[rand_idx]

                # predict
                test_predictions = np.array(clf.predict(test_trajs))

                test_accuracy = 100 * np.mean(test_predictions == test_ground_truth)

                # store values for later plotting
                train_prediction_accuracy[condition][method].append(train_accuracy)
                test_prediction_accuracy[condition][method].append(test_accuracy)

    # make plot
    for method in METHODS:
        fig, axs = plt.subplots(2, 1, facecolor=FACE_COLOR, figsize=FIG_SIZE, sharex=True, tight_layout=True)
        axs[0].hist(test_prediction_accuracy['control'][method], normed=True, color='b', lw=0)
        axs[0].hist(test_prediction_accuracy['experimental'][method], normed=True, color='g', lw=0)

        axs[1].hist(train_prediction_accuracy['control'][method], normed=True, color='b', lw=0)
        axs[1].hist(train_prediction_accuracy['experimental'][method], normed=True, color='g', lw=0)

        axs[0].legend(
            ['Training examples from same class', 'Training examples from different classes'],
            loc='best',
            fontsize=FONT_SIZE,
        )

        axs[0].set_xlabel('Test set prediction accuracy (%)')
        axs[0].set_ylabel('Probability')

        axs[1].set_xlabel('Training set prediction accuracy (%)')
        axs[1].set_ylabel('Probability')

        axs[0].set_title(
            'Experiment: {}\n {} training, {} test\n{} classifier'.format(EXPERIMENT_ID, N_TRAIN, N_TEST, method))

        for ax in axs:

            axis_tools.set_fontsize(ax, FONT_SIZE)

        fig.savefig('/Users/rkp/Desktop/classifier_{}_method_{}.png'.format(EXPERIMENT_ID, method))


if __name__ == '__main__':

    main()