from __future__ import division, print_function
import numpy as np


class GaussianLaminarPlume(object):
    """
    Simple cylindrical plume with Gaussian cross section.
    """

    def __init__(self, conc, mu, k):

        if not np.all(np.linalg.eigvals(k) > 0):

            raise ValueError('k must be positive definite')

        self.conc = conc
        self.mu = mu.flatten()[:, None]
        self.k = k
        self.k_inv = np.linalg.inv(k)

    def get_odor(self, x):
        """
        Return the concentration at a position x.

        :param x: position
        :return: odor concentration
        """

        temp = self.mu - x[1:].flatten()[:, None]

        return self.conc * np.exp(-0.5 * temp.T.dot(self.k_inv).dot(temp))


class CenterlineInferringAgent(object):
    """
    Simple tracking agent that tracks a plume based on inferring its centerline
    and is biased both toward the centerline and upwind, with the strength of
    the upwind bias increasing with the certainty around the centerline location.

    The equation for the agent's 3D motion is:

    $\tau \dot{v} = -v + \eta + b$

    where v is the velocity, \tau is a time constant, \eta is i.i.d. 3D Gaussian
    noise with identity covariance, and b is the bias term.
    """

    def __init__(self, tau, noise, bias, threshold, hit_trigger, hit_influence, k_0, k_s):

        self.tau = tau
        self.noise = noise
        self.bias = bias
        self.threshold = threshold
        self.hit_trigger = hit_trigger
        self.hit_influence = hit_influence
        self.k_0 = k_0
        self.k_s = k_s

        self.k_0_inv = np.linalg.inv(k_0)
        self.k_s_inv = np.linalg.inv(k_s)

    def get_centerline_posterior(self, hit_xs):
        """
        Get the new mean and covariance of the posterior
        :param hit_xs: array of positions where hits have occurred
        :return: mean and covariance of centerline posterior distribution
        """

        # how many hits?

        n_hits = len(hit_xs)

        # covariance first

        k_inv = self.k_0_inv + n_hits * self.k_s_inv
        k = np.linalg.inv(k_inv)

        # mean

        temp = self.k_s_inv.dot(hit_xs[:, 1:].T).sum(axis=1)
        mu = k.dot(temp)

        return mu, k

    def bias_from_centerline_distr(self, x, mu, k):
        """
        Calculate the bias term given current position and parameters of the
        centerline distribution.
        :param x: current position
        :param mu: current estimate of centerline mean
        :param k: current estimate of centerline covariance
        :return: 3D bias vector
        """

        # get crosswind component (pointing from agent to centerline estimate)

        cw = (mu - x[1:])
        cw /= np.linalg.norm(cw)

        # get upwind component

        uw = -self.hit_influence / np.sqrt(np.trace(k))

        bias = np.array([uw, cw[0], cw[1]])
        bias *= (self.bias / np.linalg.norm(bias))

        return bias

    def track(self, plume, start_pos, duration, dt):
        """
        Track a plume using the basic algorithm.

        :param plume: plume object with get_odor method
        :param threshold: odor threshold for detection
        :param start_pos: starting position
        :param duration: duration of simulation
        :param dt: numerical integration time step
        :return:
        """

        n_steps = int(duration / dt)

        centerline_mus = np.nan * np.zeros((n_steps, 2))
        centerline_ks = np.nan * np.zeros((n_steps, 2, 2))
        bs = np.nan * np.zeros((n_steps, 3))

        vs = np.nan * np.zeros((n_steps, 3))
        xs = np.nan * np.zeros((n_steps, 3))
        odors = np.nan * np.zeros((n_steps,))
        hits = np.nan * np.zeros((n_steps,))

        in_puff = False
        hit_occurred = False

        for t_ctr in range(n_steps):

            if t_ctr == 0:

                centerline_mu = np.array([0., 0])
                centerline_k = self.k_0

                b = self.bias_from_centerline_distr(
                    start_pos, centerline_mu, centerline_k)

                v = np.array([0., 0, 0])
                x = start_pos.copy()

            else:

                # get driving terms

                eta = np.random.normal(0, self.noise, (3,))
                b = self.bias_from_centerline_distr(x, centerline_mu, centerline_k)

                # update velocity and position

                v += (dt / self.tau) * (-v + eta + b)
                x += v * dt

            # sample odor

            odor = plume.get_odor(x)

            # has hit occurred?

            if self.hit_trigger == 'entry':

                if odor >= self.threshold and not in_puff:

                    hit = 1
                    in_puff = True

                else:

                    hit = 0

            elif self.hit_trigger == 'peak':

                hit = 0

                if odor >= self.threshold:

                    if odor <= last_odor and not hit_occurred:

                        hit = 1
                        hit_occurred = True

                    last_odor = odor

            if odor < self.threshold:

                last_odor = 0
                in_puff = False
                hit_occurred = False

            # store data for this time step

            centerline_mus[t_ctr] = centerline_mu
            centerline_ks[t_ctr] = centerline_k

            bs[t_ctr] = b

            vs[t_ctr] = v
            xs[t_ctr] = x

            odors[t_ctr] = odor
            hits[t_ctr] = hit

            # update centerline distribution if hit has occurred

            if hit:

                hit_xs = xs[hits == 1, :]

                centerline_mu, centerline_k = \
                    self.get_centerline_posterior(hit_xs)

        return_dict = {
            'centerline_mus': centerline_mus,
            'centerline_ks': centerline_ks,
            'bs': bs,
            'vs': vs,
            'xs': xs,
            'odors': odors,
            'hits': hits,
            'ts': np.arange(n_steps) * dt,
        }

        return return_dict
