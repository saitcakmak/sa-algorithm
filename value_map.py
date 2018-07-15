import numpy as np
import datetime
from multiprocessing import Pool as ThreadPool
from mm1_toy import queue
# from simple_quadv2 import quadv2

prob = queue

mu_gamma = 5
std_gamma = 0.1
alpha = 0.9
n0, m0 = 1000, 100


def collect_inner_samples(m, gamma, theta):
    global prob
    inner_list = []
    inner_derivative_list = []
    for j in range(m):
        val, der = prob(gamma, theta)
        inner_list.append(val)
        inner_derivative_list.append(der)
    return np.average(inner_list), np.average(inner_derivative_list)


def collect_samples(n, m, theta):
    sample_list = []
    derivative_list = []
    arg_list = []

    for i in range(n):
        gamma = np.random.randn() * std_gamma + mu_gamma
        arg_list.append((m, gamma, theta))
    pool = ThreadPool()
    results = pool.starmap(collect_inner_samples, arg_list)
    pool.close()
    pool.join()
    for res in results:
        sample_list.append(res[0])
        derivative_list.append(res[1])
    return sample_list, derivative_list


def sample():
    begin = datetime.datetime.now()
    val_list =[]
    for i in range(60):
        now = datetime.datetime.now()
        print("iter: ", i, " time: ", now-begin)
        in_list = []
        for j in range(100):
            samp, der = collect_samples(n0, m0, (5 + i/200))
            in_list.append(samp)
        val_list.append(np.average(in_list))
    return val_list


vals = sample()
np.save("mm1_vals_5to54_2xprecise_100x1000x100", vals)
