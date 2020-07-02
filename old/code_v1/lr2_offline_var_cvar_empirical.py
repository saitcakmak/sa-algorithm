"""
this code uses likelihood ratio to reduce computational budget
LR sampler distribution is adaptively chosen based on previous theta
"""
import datetime
from multiprocessing import Pool as ThreadPool
from old.code_v1.sa_params import *
from old.code_v1.mm1_toy import mm1, mm1_for_lr
import numpy as np


start = datetime.datetime.now()

string = input("enter output string: ")
prob = mm1
prob_lr = mm1_for_lr
prob_str = "mm1_lr2_" + string
delta = 0.000001


def collect_samples_empirical(m, x):
    m = int(m)
    np.random.seed()
    inner_list = np.zeros(m)
    inner_derivative_list = np.zeros(m)
    for j in range(m):
        val, der = prob(theta_hat, x)
        inner_list[j] = val
        inner_derivative_list[j] = der
    return np.average(inner_list), np.average(inner_derivative_list, 0)


def calculate_posterior(th_c, N):
    """
    return the posterior parameters
    prior is assumed gamma(2,0)
    the true distribution is exponential with theta_c
    M is the input data size
    """
    seed = np.random.random(N)
    log = np.log(seed)
    dat = (-1 / th_c) * log
    a = 2 + N
    b = np.sum(dat)
    theta_h = 1 / np.average(dat)
    return a, b, theta_h, dat


def collect_inner_samples(m, theta, x):
    """
    collect samples for a given theta for use in likelihood sampling
    """
    m = int(m)
    np.random.seed()
    out = []
    for j in range(m):
        out.append(prob_lr(theta, x))
    return out


def calc_likelihood(vals, theta_org, theta_alt):
    """
    returns the likelihood ratio of given theta's for the given data
    """
    likelihood = np.prod((np.exp(- theta_alt * (vals - delta)) - np.exp(- theta_alt * vals))
                         / (np.exp(- theta_org * (vals - delta)) - np.exp(- theta_org * vals)))
    return likelihood


def lr_estimator(samples, theta):
    lr_list = np.zeros(len(samples))
    val_sum = 0
    der_sum = 0
    for i in range(len(samples)):
        lr_list[i] = calc_likelihood(samples[i][2], samples[i][3], theta)
        val_sum = val_sum + samples[i][0] * lr_list[i]
        der_sum = der_sum + samples[i][1] * lr_list[i]
    return val_sum / np.sum(lr_list), der_sum / np.sum(lr_list), theta


def collect_samples(n, m, x):
    """
    samples thetas
    uses the previous VaR theta to draw samples to be used as LR samples
    sends these samples along with theta list to get the estimates
    returns the list of value / derivative pairs
    """
    theta_list = np.random.gamma(post_a, 1/post_b, n)
    samples = collect_inner_samples(m, theta_lr, x)
    arg_list = []
    for theta in theta_list:
        arg_list.append((samples, theta))
    pool = ThreadPool()
    results = pool.starmap(lr_estimator, arg_list)
    pool.close()
    pool.join()
    val_list = np.zeros(n)
    der_list = np.zeros(n)
    t_list = np.zeros(n)
    for i in range(n):
        val_list[i] = results[i][0]
        der_list[i] = results[i][1]
        t_list[i] = results[i][2]
    return val_list, der_list, t_list


def calc_der_var(n, m, x, alpha):
    global theta_lr
    sample_list, derivative_list, t_list = collect_samples(n, m, x)

    sort_index = np.argsort(sample_list)
    sorted_list = sample_list[sort_index]
    sorted_der = derivative_list[sort_index]
    theta_lr = t_list[sort_index][int(n*alpha)]

    return sorted_list[int(n * alpha)], sorted_der[int(n * alpha)]


def calc_der_cvar(n, m, x, alpha):
    global theta_lr
    sample_list, derivative_list, t_list = collect_samples(n, m, x)

    sort_index = np.argsort(sample_list)
    sorted_list = sample_list[sort_index]
    sorted_der = derivative_list[sort_index]
    theta_lr = t_list[sort_index][int(n*alpha)]

    return np.average(sorted_list[int(n * alpha):]), np.average(sorted_der[int(n * alpha):], 0)


def linear_budget_var(iter_count, alpha, run=0, x_0=x0, linear_coef=linear_coef0, eps_num=eps_num0, eps_denom=eps_denom0, n_m_ratio=n_m_ratio0):
    """
    start with x_0 and follow the algorithm from there
    use the iterative algorithm and map the evolution of the objective function value
    budget increases linearly as follows: n = n0+t, m = m0 + t/10 etc. The constants might change
    """
    begin = datetime.datetime.now()
    val_list = []
    der_list = []
    x_list = [x_0]
    for t in range(iter_count):
        eps = eps_num / (eps_denom + t) ** eps_power
        n = n0 + int(linear_coef * t)
        k = 1  # k in the algorithm definition
        v_list = []
        d_list = []
        for i in range(k):
            v, d = calc_der_var(n, int(n * n_m_ratio), x_list[t], alpha)
            v_list.append(v)
            d_list.append(d)
        val = np.average(v_list)
        der = np.average(d_list)
        x_next = max(np.array(x_list[t]) - eps * np.array(der), x_low)  # make sure x is not out of bounds
        x_list.append(x_next)
        val_list.append(val)
        der_list.append(der)
        now = datetime.datetime.now()
        print("run = " + str(run) + " var_ " + str(alpha) + " t = ", t, " x = ", x_list[t], " val = ", val, " der = ", der, " time: ", now-begin)
        if (t+1) % 100 == 0:
            np.save("output/" + prob_str + "_run_" + str(run) + "_VaR_" + str(alpha) + "_linear" + str(linear_coef) + "_n-m" + str(n_m_ratio) + "_t0=" + str(x_0) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_" + str(eps_power) + "_x", x_list)
        #     np.save("output/" + prob_str + "_run_" + str(run) + "_VaR_" + str(alpha) + "_linear" + str(linear_coef) + "_n-m" + str(n_m_ratio) + "_t0=" + str(x_0) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_" + str(eps_power) + "_val", val_list)
        #     np.save("output/" + prob_str + "_run_" + str(run) + "_VaR_" + str(alpha) + "_linear" + str(linear_coef) + "_n-m" + str(n_m_ratio) + "_t0=" + str(x_0) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_" + str(eps_power) + "_der", der_list)
    return x_list, val_list, der_list


def linear_budget_cvar(iter_count, alpha, run=0, x_0=x0, linear_coef=linear_coef0, eps_num=eps_num0, eps_denom=eps_denom0, n_m_ratio=n_m_ratio0):
    """
    start with x0 and follow the algorithm from there
    use the iterative algorithm and map the evolution of the objective function value
    budget increases linearly as follows: M = n0+t, m = m0 + t/10 etc. The constants might change
    """
    begin = datetime.datetime.now()
    val_list = []
    der_list = []
    x_list = [x_0]
    for t in range(iter_count):
        eps = eps_num / (eps_denom + t) ** eps_power
        n = n0 + int(linear_coef * t)
        val, der = calc_der_cvar(n, int(n * n_m_ratio), x_list[t], alpha)
        x_next = max(np.array(x_list[t]) - eps * np.array(der), x_low)  # make sure x is not out of bounds
        x_list.append(x_next)
        val_list.append(val)
        der_list.append(der)
        now = datetime.datetime.now()
        print("run = " + str(run) + " cvar_" + str(alpha) + " t = ", t, " x = ", x_list[t], " val = ", val, " der = ", der, " time: ", now-begin)
        if (t+1) % 100 == 0:
            np.save("output/" + prob_str + "_run_" + str(run) + "_CVaR_" + str(alpha) + "_linear" + str(linear_coef) + "_n-m" + str(n_m_ratio) + "_t0=" + str(x_0) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_" + str(eps_power) + "_x", x_list)
        #     np.save("output/" + prob_str + "_run_" + str(run) + "_CVaR_" + str(alpha) + "_linear" + str(linear_coef) + "_n-m" + str(n_m_ratio) + "_t0=" + str(x_0) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_" + str(eps_power) + "_val", val_list)
        #     np.save("output/" + prob_str + "_run_" + str(run) + "_CVaR_" + str(alpha) + "_linear" + str(linear_coef) + "_n-m" + str(n_m_ratio) + "_t0=" + str(x_0) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_" + str(eps_power) + "_der", der_list)
    return x_list, val_list, der_list


def linear_budget_empirical(iter_count, run=0, x_0=x0, linear_coef=linear_coef0, eps_num=eps_num0, eps_denom=eps_denom0, n_m_ratio=n_m_ratio0):
    """
    start with x_0 and follow the algorithm from there
    use the iterative algorithm and map the evolution of the objective function value
    budget increases linearly as follows: M = n0+t, m = m0 + t/10 etc. The constants might change
    """
    begin = datetime.datetime.now()
    val_list = []
    der_list = []
    x_list = [x_0]
    for t in range(iter_count):
        eps = eps_num / (eps_denom + t) ** eps_power
        n = n0 + int(linear_coef * t)
        val, der = collect_samples_empirical(int(n * n_m_ratio), x_list[t])
        x_next = max(np.array(x_list[t]) - eps * np.array(der), x_low)  # make sure x is not out of bounds
        x_list.append(x_next)
        val_list.append(val)
        der_list.append(der)
        now = datetime.datetime.now()
        print("run = " + str(run) + " empirical t = ", t, " x = ", x_list[t], " val = ", val, " der = ", der, " time: ", now-begin)
        # if (t+1) % 100 == 0:
        #     np.save("output/" + prob_str + "_run_" + str(run) + "_empirical_linear" + str(linear_coef) + "_n-m" + str(n_m_ratio) + "_t0=" + str(x_0) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_" + str(eps_power) + "_x", x_list)
        #     np.save("output/" + prob_str + "_run_" + str(run) + "_empirical_linear" + str(linear_coef) + "_n-m" + str(n_m_ratio) + "_t0=" + str(x_0) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_" + str(eps_power) + "_val", val_list)
        #     np.save("output/" + prob_str + "_run_" + str(run) + "_empirical_linear" + str(linear_coef) + "_n-m" + str(n_m_ratio) + "_t0=" + str(x_0) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_" + str(eps_power) + "_der", der_list)
    return x_list, val_list, der_list


def big_run():
    global post_a, post_b, theta_hat, data, theta_c, N
    theta_c = float(input("enter theta_c: "))
    N = int(input("enter input size N: "))
    budget = int(input("enter number of iterations: "))
    runs = int(input("enter number of runs: "))
    output = {
        "post_a_b": [],
        "theta_hat": [],
        "data": [],
        "var_0.5": [],
        # "var_0.6": [],
        "var_0.7": [],
        # "var_0.8": [],
        "var_0.9": [],
        "cvar_0.5": [],
        # "cvar_0.6": [],
        "cvar_0.7": [],
        # "cvar_0.8": [],
        "cvar_0.9": [],
        "empirical": []
        }
    for run in range(1, runs+1):
        post_a, post_b, theta_hat, data = calculate_posterior(theta_c, N)
        output["post_a_b"].append((post_a, post_b))
        output["theta_hat"].append(theta_hat)
        output["data"].append(data)
        output["var_0.5"].append(linear_budget_var(budget, 0.5, run))
        # output["var_0.6"].append(linear_budget_var(budget, 0.6, run))
        output["var_0.7"].append(linear_budget_var(budget, 0.7, run))
        # output["var_0.8"].append(linear_budget_var(budget, 0.8, run))
        output["var_0.9"].append(linear_budget_var(budget, 0.9, run))
        output["cvar_0.5"].append(linear_budget_cvar(budget, 0.5, run))
        # output["cvar_0.6"].append(linear_budget_cvar(budget, 0.6, run))
        output["cvar_0.7"].append(linear_budget_cvar(budget, 0.7, run))
        # output["cvar_0.8"].append(linear_budget_cvar(budget, 0.8, run))
        output["cvar_0.9"].append(linear_budget_cvar(budget, 0.9, run))
        output["empirical"].append(linear_budget_empirical(budget, run))
        np.save("output/combined_" + prob_str + "_N_" + str(N) + "_output.npy", output)


if __name__ == "__main__":
    N = 50
    theta_c = 10
    # post_a, post_b, theta_hat, data = calculate_posterior(theta_c, N)
    post_a = 100
    post_b = 10
    theta_hat = theta_c
    theta_lr = theta_hat
    # print(collect_samples(20, 5, 10))
    linear_budget_var(100, 0.9)
    # linear_budget_cvar(100, 0.9)

end = datetime.datetime.now()
print("time: ", end-start)
