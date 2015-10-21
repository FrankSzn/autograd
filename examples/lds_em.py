from __future__ import division
import autograd.numpy as np
from autograd import grad
from itertools import imap
import abc

# TODO deal with -1./2 factors on J's


### util

def block_array(lst):
    def helper(x, axis):
        if not isinstance(x, (list, tuple)):
            return x
        stacker = lambda x: helper(x, axis+1)
        return np.concatenate(map(stacker, x), axis=axis)
    return helper(lst, 0)


def fixed_point(f, x0):
    x1 = f(x0)
    while not same(x0, x1):
        x0, x1 = x1, f(x1)
    return x1


def same(a, b):
    if isinstance(a, np.ndarray):
        return np.allclose(a, b)
    return all(map(same, a, b))


### exponential families

class ExpFam(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def eta(theta):
        'convert theta parameters to the parameter tuple'
        pass

    @abc.abstractmethod
    def statistic(y):
        'compute the sufficient statistic tuple from data'
        pass

    @abc.abstractmethod
    def logZ(eta):
        'compute the log partition function from natural param'
        pass

    @abc.abstractmethod
    def max_likelihood(expected_stats, n):
        'compute the maximum likelihood theta parameters'
        pass


class Gaussian(ExpFam):
    @staticmethod
    def eta(theta):
        mu, Sigma = theta
        J = np.linalg.inv(Sigma)
        h = np.dot(J, mu)
        return -1./2*J, h

    @staticmethod
    def statistic(y):
        return np.outer(y,y), y

    @staticmethod
    def logZ(eta):
        J, h = -2*eta[0], eta[1]
        return -1./2 * np.dot(h, np.linalg.solve(J, h)) \
            + 1./2 * np.log(np.linalg.det(J))

    @staticmethod
    def max_likelihood(expected_stats, n):
        raise NotImplementedError  # TODO


class Regression(ExpFam):
    @staticmethod
    def eta(theta):
        raise NotImplementedError

    @staticmethod
    def statistic(y):
        raise NotImplementedError

    @staticmethod
    def logZ(eta):
        raise NotImplementedError

    @staticmethod
    def max_likelihood(expected_stats, n):
        raise NotImplementedError  # TODO


### LDS EM


def EM(init_params, data):
    eta, statistic, logZ = Gaussian.eta, Gaussian.statistic, Gaussian.logZ

    def EM_update(params):
        return M_step(E_step(params, data))

    def E_step(params, data):
        def eta_augmented(theta):
            return eta(theta) + (-logZ(eta(theta)),)

        mu_0, Sigma_0, A, BBT, C, DDT = params
        raise NotImplementedError  # TODO compute natural params

        return grad(lds_log_partition_function)(natural_params)

    def M_step(expected_stats):
        raise NotImplementedError  # TODO

    def lds_log_partition_function(natural_params):
        J0, h0, J11, J12, J22, Jnode, hnode = natural_params
        N = J0.shape[0]

        ### kalman filter steps

        def observe(Jnode, hnode):
            def propagate(Jpred, hpred):
                Jfilt, hfilt = condition_on(Jpred, hpred, Jnode, hnode)
                return predict_next(Jfilt, hfilt)
            return propagate

        def condition_on(J1, h1, J2, h2):
            return J1 + J2, h1 + h2

        def predict_next(J, h):
            big_J = block_array([[J + J11, J12], [J12.T, J22]])
            big_h = block_array([h, np.zeros_like(h)])
            big_mu, big_Sigma = info_to_mean(big_J, big_h)
            Jpredict, hpredict = mean_to_info(big_mu[N:], big_Sigma[N:, N:])

            lognorm = -1./2 * np.linalg.slogdet(J + J11)[1] \
                + 1./2 * np.dot(h, np.linalg.solve(J + J11, h)) \
                + N/2. * np.log(2*np.pi)

            return (Jpredict, hpredict), lognorm

        def info_to_mean(J, h):
            Sigma = np.linalg.inv(J)
            return np.dot(Sigma, h), Sigma


        def mean_to_info(mu, Sigma):
            J = np.linalg.inv(Sigma)
            return J, np.dot(J, mu)

        ### filtering monad

        def unit(J, h):
            return (J, h), 0.

        def bind(result, step):
            (J, h), lognorm = result
            (Jnew, hnew), term = step(J, h)
            return (Jnew, hnew), lognorm + term

        def run(result, steps):
            for step in steps:
                result = bind(result, step)
            return result

        ### running the monad

        _, lognorm = run(unit(J0, h0), imap(observe, Jnode, hnode))

        return lognorm

    return fixed_point(EM_update, init_params)
