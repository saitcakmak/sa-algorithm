import numpy as np
import scipy.stats as sci
import datetime

"""
This code estimates the c prime for the simple example
"""


def est(alpha, c):
    vals = np.zeros(c)
    interval = (1 - alpha) / c
    for i in range(c):
        t = alpha + i * (1 - alpha)/c
        z_t = sci.norm.ppf(t)
        vals[i] = z_t * sci.norm.pdf(z_t)

    estimate = 1 / (1 - alpha) * 2 / np.sqrt(5) * np.sum(vals) * interval
    return estimate


if __name__ == "__main__":
    start = datetime.datetime.now()
    print(est(0.99, 1000000))
    end = datetime.datetime.now()
    print(end-start)
