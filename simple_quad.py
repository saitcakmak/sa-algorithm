"""
the simple quadratic function is as follows:
h_gamma(theta, xi) = (theta - 10 * gamma)^2 + (gamma + theta)^2 * xi^3
the unique minimizer of the expectation for a given gamma is at theta = 10 * gamma
"""
import numpy as np


def quad(gamma, theta, seed=0):
    if seed:
        np.random.seed(seed)
    xi = np.random.randn()
    val = (theta - 10 * gamma) ** 2 + (gamma + theta) ** 2 * xi ** 3
    der = 2 * theta - 20 * gamma + 2 * (theta + gamma) * xi ** 3
    return val, der
