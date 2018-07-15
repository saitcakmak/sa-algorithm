"""
the simple quadratic function is as follows:
h_gamma(theta, xi) = 5 * (theta - 10 * gamma)^2 + log (gamma * theta) * xi
the unique minimizer of the expectation for a given gamma is at theta = 10 * gamma
"""
import numpy as np


def quadv2(gamma, theta, seed=0):
    if seed:
        np.random.seed(seed)
    xi = np.random.randn()
    val = 5 * (theta - 10 * gamma) ** 2 + np.log(gamma * theta) * xi
    der = 10 * theta - 100 * gamma + xi / theta
    return val, der
