import numpy as np
import scipy.stats as sci

dim = 2  # This is the dimension of the rvs array


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


def sampler_lr(theta, x, m):
    """
    This is not used now. Just a copy of old wrapper.
    :param theta:
    :param x:
    :param m:
    :return:
    """
    samples = np.zeros(m)
    ders = np.zeros(m)
    rvs = np.zeros((m, dim))
    likelihood = np.zeros((m, dim))

    for j in range(m):
        sample = prob(theta, x)
        samples[j] = sample[0]
        ders[j] = sample[1]
        rvs[j] = sample[2]
        # TODO: for given rvs calculate the likelihood and append it here
        likelihood[j] = sci.expon(sample[2], 1/theta) # this is just an example for exponential rv with rate theta

    return samples, ders, rvs, likelihood
