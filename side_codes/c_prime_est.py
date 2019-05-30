import numpy as np
import scipy.stats as sci
import datetime
from multiprocessing import Pool

"""
This code estimates the c prime for the simple example
"""


def est(alpha, c):
    # Don't use, this is wrong.
    vals = np.zeros(c)
    interval = (1 - alpha) / c
    for i in range(c):
        t = alpha + i * interval
        z_t = sci.norm.ppf(t)
        vals[i] = z_t * sci.norm.pdf(z_t)

    estimate = 1 / (1 - alpha) * 2 / np.sqrt(5) * np.sum(vals) * interval
    return estimate


def parallel_inner(t_start, interval, rep):
    res = 0
    for j in range(rep):
        t = t_start + j * interval
        z_t = sci.norm.ppf(t)
        res += z_t * sci.norm.pdf(z_t)
    return res


def est_parallel(alpha, c):
    # Don't use!
    interval = (1 - alpha) / c
    larger_interval = (1 - alpha) / 100
    arg_list = []
    count = 100
    for i in range(count):
        t_start = alpha + i * larger_interval
        arg_list.append((t_start, interval, int(c/count)))
    pool = Pool(count)
    pool_results = pool.starmap(parallel_inner, arg_list)
    pool.close()
    pool.join()
    estimate = 1 / (1 - alpha) * 2 / np.sqrt(5) * np.sum(pool_results) * interval
    return estimate


def mc_try(alpha, c):
    x = 2
    theta_1 = np.random.normal(0, 1, c)
    theta_2 = np.random.normal(0, 1, c)
    if alpha == 0.5:
        v_alpha = 0
    elif alpha == 0.8:
        v_alpha = 1.88192229
    elif alpha == 0.99:
        v_alpha = 5.201871986
    else:
        return -1
    val = x * theta_1 + theta_2
    indicator = val > v_alpha
    der_values = theta_1 * indicator
    estimator = (1 / (1 - alpha)) * np.average(der_values)
    return estimator


if __name__ == "__main__":
    start = datetime.datetime.now()
    # print("0.5: ", est_parallel(0.5, 100000000))
    # print("0.8: ", est_parallel(0.8, 100000000))
    # print("0.99: ", est_parallel(0.99, 100000000))
    print("0.5: ", mc_try(0.5, 100000000))
    end = datetime.datetime.now()
    print(end-start)
