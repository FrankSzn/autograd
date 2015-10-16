from __future__ import division
import abc
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd import grad


# TODO write a monad for tracking likelihoods and maybe parameter evolution?

### util

def fixed_point(f, x0):
    x1 = f(x0)
    while not same(x0, x1):
        x0, x1 = x1, f(x1)
    return x1


def same(a, b):
    if isinstance(a, np.ndarray):
        return np.allclose(a, b)
    return all(map(same, a, b))


def normalize(a):
    def replace_zeros(a):
        return np.where(a > 0., a, 1.)
    return a / replace_zeros(a.sum(-1, keepdims=True))


def inner(a, b):
    def contract(x, y):
        return np.sum(x * y)
    return sum(map(contract, a, b))


### exponential families


class ExpFam(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def eta(theta):
        'static method to convert theta parameters to the parameter tuple'
        pass

    @abc.abstractmethod
    def statistic(self, y):
        'static method to compute the sufficient statistic tuple from data'
        pass

    @abc.abstractmethod
    def logZ(self, eta):
        'static method to compute the log partition function from natural pram'
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


class NormalInverseWishart(ExpFam):
    def eta(theta):
        S, mu, nu, kappa = theta
        return np.array([S + np.outer(mu,mu) / kappa, mu/kappa, 1./kappa, nu])

    def statistic(y):
        # the NIW is the conjugate prior to the Gaussian in (mu, Sigma) params
        return Gaussian.eta(y) + (-Gaussian.logZ(Gaussian.eta(y)),)

    def logZ(eta):
        A, b, c, d = eta
        raise NotImplementedError  # TODO


### exp fam HMM EM

def EM(init_params, data, obs):
    def EM_update(params):
        return M_step(E_step(params, data))

    def E_step(params, data):
        def obs_natparam(theta):
            return obs.eta(theta) + (-obs.logZ(obs.eta(theta)),)

        pi, A, thetas = params
        natural_params = np.log(pi), np.log(A), map(obs_natparam, thetas)
        return grad(hmm_log_partition_function)(natural_params, data)

    def M_step(expected_stats):
        E_init, E_trans, E_obs_statistics = expected_stats
        pi, A = normalize(E_init), normalize(E_trans)
        thetas = map(max_likelihood, E_obs_statistics)
        return pi, A, thetas

    def hmm_log_partition_function(natural_params, data):
        log_pi, log_A, etas = natural_params
        log_alpha = log_pi
        for y in data:
            log_alpha = logsumexp(log_alpha[:,None] + log_A, axis=0) \
                + log_likelihoods(y, etas)
        return logsumexp(log_alpha)

    def log_likelihoods(y, etas):
        stat = statistic(y) + (1,)
        log_likelihood = lambda eta: inner(eta, stat)
        return np.array(map(log_likelihood, etas))

    return fixed_point(EM_update, init_params)


if __name__ == '__main__':
    np.random.seed(0)
    np.seterr(divide='ignore', invalid='raise')

    data = npr.randn(10,2)  # TODO load something interesting

    N = 2
    D = data.shape[1]

    def rand_gaussian(D):
        return npr.randn(D), np.eye(D)

    init_pi = normalize(npr.rand(N))
    init_A = normalize(npr.rand(N, N))
    init_obs_params = [rand_gaussian(D) for _ in range(N)]
    init_params = (init_pi, init_A, init_obs_params)

    pi, A, thetas = EM(init_params, data, Gaussian)
