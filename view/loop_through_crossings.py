"""
Loop through plots of plume crossings.
"""
from __future__ import print_function, division
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from db_api import models
from db_api.connect import session

ODOR_STATES = ('on', 'none', 'afterodor')
THRESHOLDS = {'fruit_fly': (0.01, 0.1), 'mosquito': (401, 410)}
DTH = 0.0001
TIMEPOINTS_BEFORE_ENTRY = 50
TIMEPOINTS_AFTER_EXIT = 50

FACE_COLOR = 'white'
FIG_SIZE = (8, 10)
LW = 2
plt.ion()


expts = session.query(models.Experiment).all()
keep_going = True
e_ctr = 0
o_ctr = 0
th_ctr = 0

fig, axs = plt.subplots(3, 1, facecolor=FACE_COLOR, figsize=FIG_SIZE,
                        tight_layout=True)
axs[2].twin = axs[2].twinx()

while keep_going:

    # get new crossing group
    expt = expts[e_ctr]
    odor_state = ODOR_STATES[o_ctr]
    th_val = THRESHOLDS[expt.insect][th_ctr]
    threshold = session.query(models.Threshold).\
        filter(models.Threshold.experiment == expt).\
        filter(models.Threshold.value.between(th_val - DTH, th_val + DTH)).first()

    crossing_group = session.query(models.CrossingGroup).\
        filter_by(experiment=expt, odor_state=odor_state, threshold=threshold).first()

    for crossing in crossing_group.crossings:
        xs = crossing.timepoint_field(session, 'position_x',
                                      first=-TIMEPOINTS_BEFORE_ENTRY,
                                      last=TIMEPOINTS_AFTER_EXIT,
                                      first_rel_to='entry',
                                      last_rel_to='exit')
        ys = crossing.timepoint_field(session, 'position_y',
                                      first=-TIMEPOINTS_BEFORE_ENTRY,
                                      last=TIMEPOINTS_AFTER_EXIT,
                                      first_rel_to='entry',
                                      last_rel_to='exit')
        zs = crossing.timepoint_field(session, 'position_z',
                                      first=-TIMEPOINTS_BEFORE_ENTRY,
                                      last=TIMEPOINTS_AFTER_EXIT,
                                      first_rel_to='entry',
                                      last_rel_to='exit')
        odors = crossing.timepoint_field(session, 'odor',
                                         first=-TIMEPOINTS_BEFORE_ENTRY,
                                         last=TIMEPOINTS_AFTER_EXIT,
                                         first_rel_to='entry',
                                         last_rel_to='exit')
        headings = crossing.timepoint_field(session, 'heading_xyz',
                                            first=-TIMEPOINTS_BEFORE_ENTRY,
                                            last=TIMEPOINTS_AFTER_EXIT,
                                            first_rel_to='entry',
                                            last_rel_to='exit')
        timesteps = crossing.timepoint_field(session, 'timestep',
                                             first=-TIMEPOINTS_BEFORE_ENTRY,
                                             last=TIMEPOINTS_AFTER_EXIT,
                                             first_rel_to='entry',
                                             last_rel_to='exit')
        entry_timestep = crossing.timepoint_field(session, 'timestep',
                                                  first=0,
                                                  last=0,
                                                  first_rel_to='entry',
                                                  last_rel_to='exit')[0]

        colors = (odors > threshold.value).astype(int)

        # plot everything
        [ax.cla() for ax in list(axs) + [axs[2].twin]]
        axs[0].scatter(xs, ys, c=colors, lw=0, cmap=cm.hot, vmin=0, vmax=2)
        axs[1].scatter(xs, zs, c=colors, lw=0, cmap=cm.hot, vmin=0, vmax=2)
        axs[2].plot(timesteps, odors, c='r', lw=LW)
        axs[2].axhline(threshold.value, c='r', ls='--', lw=LW)
        axs[2].axvline(entry_timestep, c='r', ls='--', lw=LW)
        axs[2].twin.plot(timesteps, headings, c='b', lw=LW)

        axs[0].set_xlim(-0.3, 1)
        axs[0].set_ylim(-0.15, 0.15)
        axs[1].set_xlim(-0.3, 1)
        axs[1].set_ylim(-0.15, 0.15)
        axs[2].set_xlim(timesteps[0], timesteps[-1])
        axs[2].twin.set_ylim(0, 180)

        axs[0].set_ylabel('y (m)')
        axs[1].set_ylabel('z (m)')
        axs[1].set_xlabel('x (m)')
        axs[2].set_ylabel('odor', color='r')
        axs[2].twin.set_ylabel('heading (deg)', color='b')

        axs[0].set_title('{}_{}_th{}'.format(expt.id, odor_state, threshold.value))
        plt.draw()

        command = raw_input('Command [e(xpt), o(dor_state), t(hreshold), q(uit)]?')

        if command in ['e', 'o', 't', 'q']:
            if command == 'e':
                e_ctr += 1
                e_ctr %= len(expts)
            elif command == 'o':
                o_ctr += 1
                o_ctr %= len(ODOR_STATES)
            elif command == 't':
                th_ctr += 1
                th_ctr %= len(THRESHOLDS['fruit_fly'])
            elif command == 'q':
                keep_going = False
            break
        elif command == 'pdb':
            import pdb; pdb.set_trace()