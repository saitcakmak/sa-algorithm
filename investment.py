"""
this is the investment problem as described in my notes with 3 assets for now
returns are normal with mean base + gamma * beta and std dev 0.3 * beta
the cost of capital is c1 * cap + c2 * cap^2
v2 has higher variance for higher beta, i.e, std = 0.4 * beta^2
v3 has even higher variance
"""
import numpy as np
from investment_params import b, c1, c2, base


def invest(gamma, theta, seed=0):
    if seed:
        np.random.seed(seed)
    x = np.random.multivariate_normal(base + gamma * b, 0.3 * np.diag(b))
    cap = sum(theta)
    val = c1 * cap + c2 * cap ** 2 - np.dot(theta, x)
    der = c1 + 2 * c2 * cap - x
    return val, der


def investv2(gamma, theta, seed=0):
    if seed:
        np.random.seed(seed)
    x = np.random.multivariate_normal(base + gamma * b, 0.4 * np.diag(np.square(b)))
    cap = sum(theta)
    val = c1 * cap + c2 * cap ** 2 - np.dot(theta, x)
    der = c1 + 2 * c2 * cap - x
    return val, der


def investv3(gamma, theta, seed=0):
    if seed:
        np.random.seed(seed)
    x = np.random.multivariate_normal(base + gamma * b, 0.5 * np.diag(np.power(b, 3)))
    cap = sum(theta)
    val = c1 * cap + c2 * cap ** 2 - np.dot(theta, x)
    der = c1 + 2 * c2 * cap - x
    return val, der
