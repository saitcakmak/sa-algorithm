"""
this is the investment problem as described in my notes with 3 assets for now
returns are normal with mean gamma * beta and std dev 0.3 * gamma * beta
the cost of capital is c1 * cap + c2 * cap^2
v2 has higher variance for higher beta, i.e, std = 0.4 * gamma * beta^2
"""
import numpy as np

b = np.array([1, 1.2, 0.7, 1.4, 0.85])
c1 = 0.001
c2 = 0.005


def invest(gamma, theta, seed=0):
    if seed:
        np.random.seed(seed)
    x = np.random.multivariate_normal(gamma * b, 0.3 * gamma * np.diag(b))
    cap = sum(theta)
    val = c1 * cap + c2 * cap ** 2 - np.dot(theta, x)
    der = c1 + 2 * c2 * cap - x
    return val, der


def investv2(gamma, theta, seed=0):
    if seed:
        np.random.seed(seed)
    x = np.random.multivariate_normal(gamma * b, 0.4 * gamma * np.diag(np.square(b)))
    cap = sum(theta)
    val = c1 * cap + c2 * cap ** 2 - np.dot(theta, x)
    der = c1 + 2 * c2 * cap - x
    return val, der
