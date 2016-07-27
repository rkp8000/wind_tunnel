import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plume_and_traj_3d(axs, trial, ):
    """Plot trajectory from simulation overlaid on plume."""

    axs[0].cla()
    axs[1].cla()
    # plot cross-section of plume
    axs[0].matshow(trial.pl.concxy.T, origin='lower')
    axs[1].matshow(trial.pl.concxz.T, origin='lower')

    # overlay trajectory
    x = trial.pos_idx[:, 0]
    y = trial.pos_idx[:, 1]
    z = trial.pos_idx[:, 2]

    axs[0].plot(x[:trial.ts], y[:trial.ts], color='k', lw=2)
    axs[1].plot(x[:trial.ts], z[:trial.ts], color='k', lw=2)

    # overlay hits
    if np.any(trial.detected_odor > 0):
        xhit = x[trial.detected_odor > 0]
        yhit = y[trial.detected_odor > 0]
        zhit = z[trial.detected_odor > 0]

        axs[0].scatter(xhit[:trial.ts], yhit[:trial.ts], s=50, c='r')
        axs[1].scatter(xhit[:trial.ts], zhit[:trial.ts], s=50, c='r')


def src_prob_and_traj_3d(axs, trial):
    """Plot trajectory from simulation overlaid on insect's estimate of source
    location probability distribution."""

    axs[0].cla()
    axs[1].cla()
    # plot cross-section of log probability
    axs[0].matshow(trial.ins.logprobxy.T, origin='lower')
    axs[1].matshow(trial.ins.logprobxz.T, origin='lower')

    # overlay trajectory
    x = trial.pos_idx[:, 0]
    y = trial.pos_idx[:, 1]
    z = trial.pos_idx[:, 2]

    axs[0].plot(x[:trial.ts], y[:trial.ts], color='k', lw=2)
    axs[1].plot(x[:trial.ts], z[:trial.ts], color='k', lw=2)

    # overlay hits
    if np.any(trial.detected_odor > 0):
        xhit = x[trial.detected_odor > 0]
        yhit = y[trial.detected_odor > 0]
        zhit = z[trial.detected_odor > 0]

        axs[0].scatter(xhit[:trial.ts], yhit[:trial.ts], s=50, c='r')
        axs[1].scatter(xhit[:trial.ts], zhit[:trial.ts], s=50, c='r')


def plume_traj_and_entropy_3d(axs, trial):
    """Plot trajectory from simulation overlaid on plume, along with entropy
    of source distribution as a function of time since start of search."""

    axs[0].cla()
    axs[1].cla()
    axs[2].cla()
    # plot cross-section of plume
    axs[0].matshow(trial.pl.concxy.T, origin='lower')
    axs[1].matshow(trial.pl.concxz.T, origin='lower')

    # overlay trajectory
    x = trial.pos_idx[:, 0]
    y = trial.pos_idx[:, 1]
    z = trial.pos_idx[:, 2]

    axs[0].plot(x[:trial.ts], y[:trial.ts], color='k', lw=2)
    axs[1].plot(x[:trial.ts], z[:trial.ts], color='k', lw=2)

    # plot entropy
    ts = np.arange(len(x))
    axs[2].plot(ts[:trial.ts], trial.entropies[:trial.ts], lw=2)

    # overlay hits
    if np.any(trial.detected_odor > 0):
        xhit = x[trial.detected_odor > 0]
        yhit = y[trial.detected_odor > 0]
        zhit = z[trial.detected_odor > 0]

        ts_hit = ts[trial.detected_odor > 0]
        entropies_hit = trial.entropies[trial.detected_odor > 0]

        axs[0].scatter(xhit[:trial.ts], yhit[:trial.ts], s=50, c='r')
        axs[1].scatter(xhit[:trial.ts], zhit[:trial.ts], s=50, c='r')
        axs[2].scatter(ts_hit[:trial.ts], entropies_hit[:trial.ts], s=50, c='r')

    # label axes
    axs[2].set_xlabel('time step')
    axs[2].set_ylabel('source position entropy')


def multi_traj_3d(axs, env, bkgd, trajs, hits=None, colors=None):
    """Plot multiple trajectories in 3d overlaid on one another."""

    [ax.cla() for ax in axs]

    if hits is None:
        hits = []

    # get axis limits and extent from env
    xlim = [env.xbins[0], env.xbins[-1]]
    ylim = [env.ybins[0], env.ybins[-1]]
    zlim = [env.zbins[0], env.zbins[-1]]

    extent_xy = xlim + ylim
    extent_xz = xlim + zlim

    axs[0].matshow(bkgd[0].T, origin='lower', extent=extent_xy)
    axs[1].matshow(bkgd[1].T, origin='lower', extent=extent_xz)

    axs[0].set_xlim(xlim)
    axs[0].set_ylim(ylim)

    axs[1].set_xlim(xlim)
    axs[1].set_ylim(zlim)

    if not colors:
        colors = ['k', 'g', 'r', 'b']
    hit_colors = ['r', 'c', 'm', 'y']

    for t_ctr, traj in enumerate(trajs):
        axs[0].plot(traj[:, 0], traj[:, 1], c=colors[t_ctr], lw=2)
        axs[1].plot(traj[:, 0], traj[:, 2], c=colors[t_ctr], lw=2)

    for t_ctr, hit in enumerate(hits):
        traj = trajs[t_ctr]
        axs[0].scatter(traj[hit > 0, 0], traj[hit > 0, 1], s=50,
                       c=hit_colors[t_ctr], marker='x', lw=2, zorder=10)
        axs[1].scatter(traj[hit > 0, 0], traj[hit > 0, 2], s=50,
                       c=hit_colors[t_ctr], marker='x', lw=2, zorder=10)


def multi_traj_3d_with_entropy(axs, env, bkgd, trajs, entropies, hits=None, colors=None):
    """Plot multiple trajectories in 3d overlaid on one another. Also plot their entropy
        time-series."""

    multi_traj_3d(axs, env, bkgd, trajs, hits=hits, colors=colors)

    if not colors:
        colors = ['k', 'g', 'r', 'b']
    hit_colors = ['r', 'c', 'm', 'y']

    # plot entropy
    for t_ctr, entropy in enumerate(entropies):
        axs[2].plot(entropy, lw=2, color=colors[t_ctr])

    axs[2].set_xlabel('time steps')
    axs[2].set_ylabel('source position entropy')


def traj_3d_with_segments_and_entropy(axs, env, bkgd, traj, seg_starts, seg_ends, entropies, colors=None):
    """
    Plot a single trajectory with its segments highlighted and its entropy plotted below.
    """

    [ax.cla() for ax in axs]

    # get axis limits and extent from env
    xlim = [env.xbins[0], env.xbins[-1]]
    ylim = [env.ybins[0], env.ybins[-1]]
    zlim = [env.zbins[0], env.zbins[-1]]

    extent_xy = xlim + ylim
    extent_xz = xlim + zlim

    axs[0].matshow(bkgd[0].T, origin='lower', extent=extent_xy, zorder=0, cmap=cm.hot)
    axs[1].matshow(bkgd[1].T, origin='lower', extent=extent_xz, zorder=0, cmap=cm.hot)

    axs[0].set_xlim(xlim)
    axs[0].set_ylim(ylim)

    axs[1].set_xlim(xlim)
    axs[1].set_ylim(zlim)

    # plot trajectory
    axs[0].plot(traj[:, 0], traj[:, 1], c='w', lw=2, zorder=1)
    axs[1].plot(traj[:, 0], traj[:, 2], c='w', lw=2, zorder=1)

    # plot entropy
    t = np.arange(len(entropies))
    axs[2].plot(t, entropies, c='k', lw=2)

    # plot segments overlaid
    for seg_start, seg_end in zip(seg_starts, seg_ends):
        axs[0].plot(traj[seg_start:seg_end, 0],
                    traj[seg_start:seg_end, 1],
                    c='c', lw=2, zorder=2)
        axs[1].plot(traj[seg_start:seg_end, 0],
                    traj[seg_start:seg_end, 2],
                    c='c', lw=2, zorder=2)
        axs[2].plot(t[seg_start:seg_end], entropies[seg_start:seg_end], c='c', lw=2)

    axs[2].set_xlim(0, len(entropies))
    axs[2].set_xlabel('time steps')
    axs[2].set_ylabel('source position entropy')