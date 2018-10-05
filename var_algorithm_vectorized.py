import numpy as np
import datetime
from multiprocessing import Pool as ThreadPool
from mm1_toy import queue
from simple_quad import quad, quadv2, quadv3
from prod_inv import prod
from investment import invest, investv2, investv3, investv4
from gamma_params import *
from sa_params import *

start = datetime.datetime.now()

prob = investv4
prob_str = "investv4"


def collect_inner_samples(m, gamma, theta):
    global prob
    inner_list = []
    inner_derivative_list = []
    for j in range(m):
        val, der = prob(gamma, theta)
        inner_list.append(val)
        inner_derivative_list.append(der)
    return np.average(inner_list), np.average(inner_derivative_list, 0)


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
    return np.array(sample_list), np.array(derivative_list)


def calc_der(n, m, theta):
    sample_list, derivative_list = collect_samples(n, m, theta)

    sort_index = np.argsort(sample_list)
    sorted_list = sample_list[sort_index]
    sorted_der = derivative_list[sort_index]

    return sorted_list[int(np.ceil(n * alpha))], sorted_der[int(np.ceil(n * alpha))]


def fixed_budget(iter_count, t0=theta0, mult=10, eps_num=eps_num0, eps_denom=eps_denom0, n_m_ratio=n_m_ratio0):
    """
    start with mu0 and follow the algorithm from there
    use the iterative algorithm and map the evolution of the objective function value
    :return:
    """
    begin = datetime.datetime.now()
    val_list = []
    der_list = []
    eps_list = []
    theta_list = [t0]
    n = mult * n0
    for k in range(iter_count):
        eps = eps_num / (eps_denom + k) ** eps_power
        val, der = calc_der(n, int(n * n_m_ratio), theta_list[k])
        theta_next = max(np.array(theta_list[k]) - eps * np.array(der), t_low)  # make sure theta is not out of bounds
        theta_list.append(theta_next)
        val_list.append(val)
        der_list.append(der)
        eps_list.append(eps)
        now = datetime.datetime.now()
        print("k = ", k, " theta = ", theta_list[k], " val = ", val, " der = ", der, " time: ", now-begin)
        if k % 100 == 0:
            np.save(prob_str + "_VaR" + "_" + str(mult) + "xfixed_n-m" + str(n_m_ratio) + "_t0=" + str(t0) + "_mu" + str(mu_gamma) + "_std" + str(std_gamma) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_" + str(eps_power) + "_iter_" + str(k) + "_theta", theta_list)
            np.save(prob_str + "_VaR" + "_" + str(mult) + "xfixed_n-m" + str(n_m_ratio) + "_t0=" + str(t0) + "_mu" + str(mu_gamma) + "_std" + str(std_gamma) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_" + str(eps_power) + "_iter_" + str(k) + "_val", val_list)
            np.save(prob_str + "_VaR" + "_" + str(mult) + "xfixed_n-m" + str(n_m_ratio) + "_t0=" + str(t0) + "_mu" + str(mu_gamma) + "_std" + str(std_gamma) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_" + str(eps_power) + "_iter_" + str(k) + "_der", der_list)
    return theta_list, val_list, der_list, eps_list


def linear_budget(iter_count, t0=theta0, linear_coef=linear_coef0, eps_num=eps_num0, eps_denom=eps_denom0, n_m_ratio=n_m_ratio0):
    """
    start with mu0 and follow the algorithm from there
    use the iterative algorithm and map the evolution of the objective function value
    budget increases linearly as follows: n = n0+k, m = m0 + k/10 etc. The constants might change
    :return:
    """
    begin = datetime.datetime.now()
    val_list = []
    der_list = []
    eps_list = []
    theta_list = [t0]
    for k in range(iter_count):
        eps = eps_num / (eps_denom + k) ** eps_power
        n = n0 + int(linear_coef * k)
        val, der = calc_der(n, int(n * n_m_ratio), theta_list[k])
        theta_next = max(np.array(theta_list[k]) - eps * np.array(der), t_low)  # make sure theta is not out of bounds
        theta_list.append(theta_next)
        val_list.append(val)
        der_list.append(der)
        eps_list.append(eps)
        now = datetime.datetime.now()
        print("k = ", k, " theta = ", theta_list[k], " val = ", val, " der = ", der, " time: ", now-begin)
        if k % 100 == 0:
            np.save(prob_str + "_VaR" + "_linear" + str(linear_coef) + "_n-m" + str(n_m_ratio) + "_t0=" + str(t0) + "_mu" + str(mu_gamma) + "_std" + str(std_gamma) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_" + str(eps_power) + "_iter_" + str(k) + "_theta", theta_list)
            np.save(prob_str + "_VaR" + "_linear" + str(linear_coef) + "_n-m" + str(n_m_ratio) + "_t0=" + str(t0) + "_mu" + str(mu_gamma) + "_std" + str(std_gamma) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_" + str(eps_power) + "_iter_" + str(k) + "_val", val_list)
            np.save(prob_str + "_VaR" + "_linear" + str(linear_coef) + "_n-m" + str(n_m_ratio) + "_t0=" + str(t0) + "_mu" + str(mu_gamma) + "_std" + str(std_gamma) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_" + str(eps_power) + "_iter_" + str(k) + "_der", der_list)
    return theta_list, val_list, der_list, eps_list


if __name__ == "__main__":
    linear_budget(2501)

end = datetime.datetime.now()
print("time: ", end-start)
