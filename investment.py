"""
this is the investment problem as described in my notes with 3 assets for now
returns are normal with mean base + gamma * beta and std dev 0.3 * beta
the cost of capital is c1 * cap + c2 * cap^2
v2 has higher variance for higher beta, i.e, std = 0.4 * beta^2
v3 has even higher variance
"""
import numpy as np

b = np.array([1, 1.2, 0.7, 1.4, 0.85])
c1 = 0.001
c2 = 0.005
base = 0.04


def invest(gamma, theta, seed=0):
    if seed:
        np.random.seed(seed)
    x = np.random.multivariate_normal(0, np.diag(b)) * 0.3 + base + gamma * b
    cap = sum(theta)
    val = c1 * cap + c2 * cap ** 2 - np.dot(theta, x)
    der = c1 + 2 * c2 * cap - x
    return val, der


def investv2(gamma, theta, seed=0):
    if seed:
        np.random.seed(seed)
    x = np.random.multivariate_normal(0, np.diag(np.square(b))) * 0.4 + base + gamma * b
    cap = sum(theta)
    val = c1 * cap + c2 * cap ** 2 - np.dot(theta, x)
    der = c1 + 2 * c2 * cap - x
    return val, der


def investv3(gamma, theta, seed=0):
    if seed:
        np.random.seed(seed)
    x = np.random.multivariate_normal(0, np.diag(np.power(b, 3))) * 0.5 + base + gamma * b
    cap = sum(theta)
    val = c1 * cap + c2 * cap ** 2 - np.dot(theta, x)
    der = c1 + 2 * c2 * cap - x
    return val, der
