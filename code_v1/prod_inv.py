"""
this is an implementation of production inventory problem from Hong2009VaR
where d_i follows exponential with unknown rate
"""
import numpy as np

"""
this code behaves weird!!
"""

c = 1
h = 0.1
b = 0.2
M = 20
r = [0.0]


def prod(theta, x, seed=0):
    inv = [x]
    val, der = 0, 0
    if seed:
        np.random.seed(seed)
    seed = np.random.random(M)
    log = np.log(seed)
    d = (-1 / theta) * log
    for i in range(M):
        inv.append(inv[i] - d[i] + r[i])
        r.append(min(c, max(x + d[i] - (inv[i] + r[i]), 0)))
        val += h * (r[i] + max(inv[i], 0)) - b * min(inv[i], 0)
        der += (inv[i] > 0) * h - (inv[i] < 0) * b
    return val, der
