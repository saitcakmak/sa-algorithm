import numpy as np
import scipy.stats as sci
import two_sided


def simple_sampler(theta, x, m):
    xi_0 = theta[0] + np.random.normal(0, 1, m)
    xi_1 = theta[1] + np.random.normal(0, 1, m)

    samples = x * xi_0 + xi_1
    ders = xi_0
    return samples, ders


def simple_sampler_lr(theta, x, m):
    xi_0 = theta[0] + np.random.normal(0, 1, m)
    xi_1 = theta[1] + np.random.normal(0, 1, m)

    samples = x * xi_0 + xi_1
    ders = xi_0
    likelihood_0 = sci.norm.pdf(xi_0 - theta[0])
    likelihood_1 = sci.norm.pdf(xi_1 - theta[1])
    rvs = np.transpose([xi_0, xi_1])
    likelihood = np.transpose([likelihood_0, likelihood_1])
    return samples, ders, rvs, likelihood


def quad_sampler(theta, x, m):
    """ in this problem, exponential is with scale theta, rate 1/theta"""
    xi = np.random.exponential(theta, m)
    temp = x - xi
    samples = temp ** 2
    ders = 2 * x * temp
    return samples, ders


def quad_sampler_lr(theta, x, m):
    """ in this problem, exponential is with scale theta, rate 1/theta"""
    xi = np.random.exponential(theta, m)
    temp = x - xi
    samples = temp ** 2
    ders = 2 * x * temp
    rvs = xi
    likelihood = (1 / theta) * np.exp((- xi / theta))
    return samples, ders, rvs, likelihood


def two_sided_sampler(theta, x, m):
    samples = np.zeros(m)
    ders = np.zeros(m)
    for i in range(m):
        run = two_sided.two_sided(theta, x)
        samples[i] = run[0]
        ders[i] = run[1]
    return samples, ders


def two_sided_sampler_lr(theta, x, m):
    dim = 200
    samples = np.zeros(m)
    ders = np.zeros(m)
    rvs = np.zeros((m, dim))
    likelihood = np.zeros((m, dim))
    for i in range(m):
        run = two_sided.two_sided_lr(theta, x)
        samples[i] = run[0]
        ders[i] = run[1]
        rvs[i] = run[2]
        likelihood[i] = run[3]
    return samples, ders, rvs, likelihood


def simple_normal_sampler(theta, x, m):
    r"""
    For sampling from the simple normal example where h(x, \xi) = x \theta_1 + x^2 \theta_2 + x \xi
    :param theta:
    :param x:
    :param m:
    :return:
    """
    if theta.ndim > 1:
        raise ValueError("Only handles 1 theta at a time!")
    xi = np.random.normal(0, theta[0] ** 2, m)
    samples = x * theta[0] + x ** 2 * theta[1] + x * xi
    ders = theta[0] + 2 * x * theta[1] + xi
    return samples, ders
