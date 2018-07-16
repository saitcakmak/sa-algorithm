import numpy as np
import datetime
from multiprocessing import Pool as ThreadPool
from mm1_toy import queue
from simple_quadv2 import quadv2
from prod_inv import prod

prob = prod
prob_str = "prod"

mu_gamma = 1
std_gamma = 0.1
alpha = 0.9
rep, n0, m0 = 40, 4000, 400
t0 = 5
length, precision = 50, 10


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
    for i in range(length):
        in_list = []
        for j in range(10):
            now = datetime.datetime.now()
            print("i: ", i, " j: ", j, " time: ", now - begin)
            samp, der = collect_samples(n0, m0, (t0 + i/precision))
            in_list.append(samp)
        val_list.append(np.average(in_list))
    return val_list


vals = sample()
np.save(prob_str + "_vals_mu" + str(mu_gamma) + "_std" + str(std_gamma) + "_" + str(t0) + "to" + str(t0 + length/precision) + "_" + str(rep) + "x" + str(n0) + "x" + str(m0), vals)
