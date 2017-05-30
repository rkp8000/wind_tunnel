from __future__ import division, print_function
import numpy as np


class GaussianLaminarPlume(object):
    """
    Simple cylindrical plume with Gaussian cross section.
    """

    def __init__(self, conc, mu, std):

        self.conc = conc
        self.mu = mu.flatten()[:, None]
        self.k = np.array([[std[0] ** 2, 0.], [0, std[1] ** 2]])
        self.k_inv = np.linalg.inv(self.k)

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

    def __init__(
            self, tau, noise, bias, threshold,
            hit_trigger, hit_influence, k_0, k_s,
            tau_memory, bounds):

        self.tau = tau
        self.noise = noise
        self.bias = bias
        self.threshold = threshold
        self.hit_trigger = hit_trigger
        self.hit_influence = hit_influence
        self.k_0 = k_0
        self.k_s = k_s
        self.tau_memory = tau_memory

        self.k_0_inv = np.linalg.inv(k_0)
        self.k_s_inv = np.linalg.inv(k_s)

        self.bounds = bounds

    def update_centerline_posterior(self, centerline_mu, centerline_k, hit_x):
        """
        Get the new mean and covariance of the posterior
        :param centerline_mu: previous mean
        :param centerline_k: previous covariance
        :param hit_x: position where hit occurred
        :return: mean and covariance of centerline posterior distribution
        """

        # covariance first

        k_inv_prev = np.linalg.inv(centerline_k)
        k_inv = k_inv_prev + self.k_s_inv
        k = np.linalg.inv(k_inv)

        # mean

        temp = self.k_s_inv.dot(hit_x[1:]) + k_inv_prev.dot(centerline_mu)
        mu = k.dot(temp)

        return mu, k

    def decay_centerline_posterior(self, centerline_mu, centerline_k, dt):
        """
        Decay the centerline posterior using the memory time constant self.tau_memory.
        :param centerline_mu: previous mean
        :param centerline_k: previous covariance
        :param dt: simulation time step
        :return: decayed mu and k
        """

        d_centerline_mu = (dt / self.tau_memory) * (-centerline_mu)
        d_centerline_k = (dt / self.tau_memory) * (-centerline_k + self.k_0)

        return centerline_mu + d_centerline_mu, centerline_k + d_centerline_k

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

        # get certainty
        certainty = np.linalg.det(np.linalg.inv(k))

        # get bias from certainty
        uw = -self.hit_influence * certainty
        bias = np.array([uw, cw[0], cw[1]])

        return bias

    def reflect_if_out_of_bounds(self, v, x):
        """
        Check if a position is within the bounds
        :param v: current velocity
        :param x: current position
        :return: v, x corrected if x was out of bounds
        """

        if self.bounds is None:

            return v, x

        else:

            v_new = v.copy()
            x_new = x.copy()

            for dim in range(3):

                if x[dim] < self.bounds[dim][0]:

                    v_new[dim] *= -1
                    x_new[dim] = 2 * self.bounds[dim][0] - x[dim]

                elif x[dim] > self.bounds[dim][1]:

                    v_new[dim] *= -1
                    x_new[dim] = 2 * self.bounds[dim][1] - x[dim]

            return v_new, x_new

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

        last_odor = 0
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

            v, x = self.reflect_if_out_of_bounds(v, x)

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
                # sharpen posterior if hit occurred
                centerline_mu, centerline_k = \
                    self.update_centerline_posterior(centerline_mu, centerline_k, x)
            else:
                # otherwise let posterior decay back towards prior
                centerline_mu, centerline_k = \
                    self.decay_centerline_posterior(centerline_mu, centerline_k, dt)

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


class SurgingAgent(object):
    """
    Tracking agent model with same base dynamics as Centerline-inferring model
    except that plume-crossing triggers a brief upwind surge, and there is
    no memory across multiple crossings.
    """

    def __init__(
            self, tau, noise, bias, threshold, hit_trigger,
            surge_amp, tau_surge, bounds):

        self.tau = tau
        self.noise = noise
        self.bias = bias
        self.threshold = threshold
        self.hit_trigger = hit_trigger

        self.surge_amp = surge_amp
        self.tau_surge = tau_surge

        self.surge_amp_ = surge_amp / (tau_surge * np.exp(-1))

        self.bounds = bounds

    def reflect_if_out_of_bounds(self, v, x):
        """
        Check if a position is within the bounds
        :param v: current velocity
        :param x: current position
        :return: v, x corrected if x was out of bounds
        """

        if self.bounds is None:

            return v, x

        else:

            v_new = v.copy()
            x_new = x.copy()

            for dim in range(3):

                if x[dim] < self.bounds[dim][0]:

                    v_new[dim] *= -1
                    x_new[dim] = 2 * self.bounds[dim][0] - x[dim]

                elif x[dim] > self.bounds[dim][1]:

                    v_new[dim] *= -1
                    x_new[dim] = 2 * self.bounds[dim][1] - x[dim]

            return v_new, x_new

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
        ts = np.arange(n_steps) * dt

        bs = np.nan * np.zeros((n_steps, 3))
        surges = np.zeros(n_steps)

        vs = np.nan * np.zeros((n_steps, 3))
        xs = np.nan * np.zeros((n_steps, 3))
        odors = np.nan * np.zeros((n_steps,))
        hits = np.nan * np.zeros((n_steps,))

        last_odor = 0
        in_puff = False
        hit_occurred = False

        for t_ctr in range(n_steps):

            if t_ctr == 0:

                b = np.array([0, -start_pos[1], -start_pos[2]])
                b *= (self.bias / np.linalg.norm(b))

                v = np.array([0., 0, 0])
                x = start_pos.copy()

            else:

                # get driving terms
                eta = np.random.normal(0, self.noise, (3,))
                b = np.array([0, -x[1], -x[2]])
                b *= (self.bias / np.linalg.norm(b))

                # update velocity and position
                v += (dt / self.tau) * (-v + eta + b - surges[t_ctr])
                x += v * dt

            v, x = self.reflect_if_out_of_bounds(v, x)

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

            bs[t_ctr] = b

            vs[t_ctr] = v
            xs[t_ctr] = x

            odors[t_ctr] = odor
            hits[t_ctr] = hit

            # add new surge force if hit has occurred
            if hit:
                # compute alpha function starting at current time point
                # and add it to surges array
                ts_ = ts[t_ctr:] - ts[t_ctr]
                surges[t_ctr:] += (self.surge_amp_*ts_*np.exp(-ts_/self.tau_surge))

        return_dict = {
            'surges': surges,
            'bs': bs,
            'vs': vs,
            'xs': xs,
            'odors': odors,
            'hits': hits,
            'ts': ts,
        }

        return return_dict

