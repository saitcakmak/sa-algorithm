"""
this is an implementation of production inventory problem from Hong2009VaR
where d_i follows exponential with rate gamma
"""
import numpy as np

c = 0.5
h = 0.1
b = 0.2
n = 20
r = [0.0]


def prod(gamma, theta, seed=0):
    inv = [theta]
    val, der = 0, 0
    if seed:
        np.random.seed(seed)
    d = np.random.exponential(1/gamma, 20)
    for i in range(n):
        inv.append(inv[i] - d[i] + r[i])
        r.append(min(c, max(theta + d[i] - (inv[i] + r[i]), 0)))
        val += h * (r[i] + max(inv[i], 0)) - b * min(inv[i], 0)
        der += (inv[i] > 0) * h - (inv[i] < 0) * b
    return val, der
