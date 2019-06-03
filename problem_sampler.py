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
