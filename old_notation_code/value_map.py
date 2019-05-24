import numpy as np
import datetime
from multiprocessing import Pool as ThreadPool
from code_v1.prod_inv import prod

prob = prod
prob_str = "prod"

mu_gamma = 1
std_gamma = 0.1
alpha = 0.9
rep0, n0, m0 = 10, 100, 100
t0 = 5
length0, precision0 = 50, 10


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


def calc_der(n, m, theta):
    sample_list, derivative_list = collect_samples(n, m, theta)

    var_alpha = np.sort(sample_list)[int(n * alpha)]

    cvar_list = []
    cvar_der_list = []
    for i in range(n):
        if sample_list[i] >= var_alpha:
            cvar_list.append(sample_list[i])
            cvar_der_list.append(derivative_list[i])

    return np.average(cvar_list), np.average(cvar_der_list)


def sample(n=n0, m=m0, rep=rep0, length=length0, precision=precision0, t=t0):
    begin = datetime.datetime.now()
    val_list = []
    der_list = []
    for i in range(length):
        in_list = []
        in_der_list = []
        for j in range(rep):
            now = datetime.datetime.now()
            print("i: ", i, " j: ", j, " time: ", now - begin)
            samp, der = calc_der(n, m, (t + i/precision))
            in_list.append(samp)
            in_der_list.append(der)
        val_list.append(np.average(in_list))
        der_list.append(np.average(in_der_list))
    np.save(prob_str + "_fixed_vals_mu" + str(mu_gamma) + "_std" + str(std_gamma) + "_" + str(t) + "to" + str(
        t0 + length / precision) + "_" + str(rep) + "x" + str(n) + "x" + str(m), val_list)
    np.save(prob_str + "_fixed_der_mu" + str(mu_gamma) + "_std" + str(std_gamma) + "_" + str(t) + "to" + str(
        t0 + length / precision) + "_" + str(rep) + "x" + str(n) + "x" + str(m), der_list)
    return val_list, der_list
