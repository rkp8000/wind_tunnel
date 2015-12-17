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

ODOR_STATES = {'on', 'none'}

INTEGRATED_ODOR_THRESHOLDS = {
    'on': 30,
    'none': 30,
}

N_TRAIN = 50
N_TEST = 100
N_TRIALS = 200

FACE_COLOR = 'white'
FIG_SIZE = (10, 12)
FONT_SIZE = 20


def main():

    train_prediction_accuracy = {}
    test_prediction_accuracy = {}

    for condition in ['control', 'experimental']:

        print('Condition: {}'.format(condition))

        train_prediction_accuracy[condition] = []
        test_prediction_accuracy[condition] = []

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

        assert len(trajs['on']) >= N_TRAIN + N_TEST
        assert len(trajs['none']) >= N_TRAIN + N_TEST

        for tr_ctr in range(N_TRIALS):

            print('Trial # {}'.format(tr_ctr))

            vels = {}

            # get all data
            for odor_state in ODOR_STATES:

                shuffle(trajs[odor_state])

                vels[odor_state] = {
                    'train': [traj.velocities(session) for traj in trajs[odor_state][:N_TRAIN]],
                    'test': [traj.velocities(session) for traj in trajs[odor_state][N_TRAIN:N_TRAIN+N_TEST]],
                }

            # train a classifer
            clf = tsc.VarClassifierBinary(dim=3, order=2)
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

            test_predictions = np.array(clf.predict(test_trajs))

            test_accuracy = 100 * np.mean(test_predictions == test_ground_truth)

            #print('{}% positive guesses training'.format(100 * np.mean(train_predictions == 1)))
            #print('{}% correct training.'.format(train_accuracy))
            #print('{}% positive guesses test'.format(100 * np.mean(test_predictions == 1)))
            #print('{}% correct test.'.format(test_accuracy))

            train_prediction_accuracy[condition].append(train_accuracy)
            test_prediction_accuracy[condition].append(test_accuracy)

    # make plot
    fig, axs = plt.subplots(2, 1, facecolor=FACE_COLOR, figsize=FIG_SIZE, sharex=True, tight_layout=True)
    axs[0].hist(test_prediction_accuracy['control'], normed=True, color='b', lw=0)
    axs[0].hist(test_prediction_accuracy['experimental'], normed=True, color='g', lw=0)

    axs[1].hist(train_prediction_accuracy['control'], normed=True, color='b', lw=0)
    axs[1].hist(train_prediction_accuracy['experimental'], normed=True, color='g', lw=0)

    axs[0].legend(
        ['Training examples from same class', 'Training examples from different classes'],
        loc='best',
        fontsize=FONT_SIZE,
    )

    axs[0].set_xlabel('Test set prediction accuracy (%)')
    axs[0].set_ylabel('Probability')

    axs[1].set_xlabel('Training set prediction accuracy (%)')
    axs[1].set_ylabel('Probability')

    axs[0].set_title('Experiment: {}\n {} training, {} test'.format(EXPERIMENT_ID, N_TRAIN, N_TEST))

    for ax in axs:
        axis_tools.set_fontsize(ax, FONT_SIZE)

    plt.show()


if __name__ == '__main__':
    main()