import numpy as np
import datetime
from multiprocessing import Pool as ThreadPool
from mm1_toy import queue
from simple_quad import quad
from simple_quadv2 import quadv2
from mm1_toy5x import queue5x
from simple_quadv3 import quadv3
from prod_inv import prod
start = datetime.datetime.now()

prob = prod
prob_str = "prod"

mu_gamma = 1
std_gamma = 0.1
alpha = 0.9

theta0 = 3
n0, m0 = 100, 10

# step size epsilon is similar to that of gasso
eps_num = 1
eps_denom = 100
eps_power = 0.6
linear_coef = 1


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


def fixed_budget(iter_count, t0=theta0, mult=4):
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
    for k in range(iter_count):
        eps = eps_num / (eps_denom + k) ** eps_power
        val, der = calc_der(mult * n0, mult * m0, theta_list[k])
        theta_next = theta_list[k] - eps * der
        theta_list.append(theta_next)
        val_list.append(val)
        der_list.append(der)
        eps_list.append(eps)
        now = datetime.datetime.now()
        print("k = ", k, " theta = ", theta_list[k], " val = ", val, " der = ", der, " time: ", now-begin)
        if k % 100 == 0:
            np.save(prob_str + "_" + str(mult) + "xfixed_t0=" + str(t0) + "_mu" + str(mu_gamma) + "_std" + str(std_gamma) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_iter_" + str(k) + "_theta", theta_list)
            np.save(prob_str + "_" + str(mult) + "xfixed_t0=" + str(t0) + "_mu" + str(mu_gamma) + "_std" + str(std_gamma) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_iter_" + str(k) + "_val", val_list)
            np.save(prob_str + "_" + str(mult) + "xfixed_t0=" + str(t0) + "_mu" + str(mu_gamma) + "_std" + str(std_gamma) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_iter_" + str(k) + "_der", der_list)
    return theta_list, val_list, der_list, eps_list


def linear_budget(iter_count, t0=theta0):
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
        val, der = calc_der(n0 + int(linear_coef * k), m0 + int(linear_coef * k/10), theta_list[k])
        theta_next = theta_list[k] - eps * der
        theta_list.append(theta_next)
        val_list.append(val)
        der_list.append(der)
        eps_list.append(eps)
        now = datetime.datetime.now()
        print("k = ", k, " theta = ", theta_list[k], " val = ", val, " der = ", der, " time: ", now-begin)
        if k % 100 == 0:
            np.save(prob_str + "_linear" + str(linear_coef) + "_t0=" + str(t0) + "_mu" + str(mu_gamma) + "_std" + str(std_gamma) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_iter_" + str(k) + "_theta", theta_list)
            np.save(prob_str + "_linear" + str(linear_coef) + "_t0=" + str(t0) + "_mu" + str(mu_gamma) + "_std" + str(std_gamma) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_iter_" + str(k) + "_val", val_list)
            np.save(prob_str + "_linear" + str(linear_coef) + "_t0=" + str(t0) + "_mu" + str(mu_gamma) + "_std" + str(std_gamma) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_iter_" + str(k) + "_der", der_list)
    return theta_list, val_list, der_list, eps_list


def dynamic_step_linear_budget(iter_count, t0=theta0):
    """
    start with mu0 and follow the algorithm from there
    use the iterative algorithm and map the evolution of the objective function value
    budget increases linearly as follows: n = n0+k, m = m0 + k/10 etc. The constants might change
    the step size only decreases if the sign of derivative changes
    :return:
    """
    begin = datetime.datetime.now()
    val_list = []
    der_list = []
    eps_list = []
    theta_list = [t0]
    s = 0
    for k in range(iter_count):
        eps = eps_num / (eps_denom + s) ** eps_power
        val, der = calc_der(n0 + int(linear_coef * k), m0 + int(linear_coef * k/10), theta_list[k])
        theta_next = theta_list[k] - eps * der
        theta_list.append(theta_next)
        val_list.append(val)
        der_list.append(der)
        eps_list.append(eps)
        now = datetime.datetime.now()
        print("k = ", k, " theta = ", theta_list[k], " val = ", val, " der = ", der, " eps = ", eps, " time: ", now-begin)
        if k % 100 == 0:
            np.save(prob_str + "_dynamic_linear" + str(linear_coef) + "_t0=" + str(t0) + "_mu" + str(mu_gamma) + "_std" + str(std_gamma) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_iter_" + str(k) + "_theta", theta_list)
            np.save(prob_str + "_dynamic_linear" + str(linear_coef) + "_t0=" + str(t0) + "_mu" + str(mu_gamma) + "_std" + str(std_gamma) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_iter_" + str(k) + "_val", val_list)
            np.save(prob_str + "_dynamic_linear" + str(linear_coef) + "_t0=" + str(t0) + "_mu" + str(mu_gamma) + "_std" + str(std_gamma) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_iter_" + str(k) + "_der", der_list)
        if der_list[k] * der_list[max(k-1, 0)] < 0:
            s += 1
    return theta_list, val_list, der_list, eps_list


# dynamic_step_linear_budget(50)

end = datetime.datetime.now()
print("time: ", end-start)
