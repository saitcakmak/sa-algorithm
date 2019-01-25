import datetime
import numpy as np
from sa_params import *


start = datetime.datetime.now()

string = input("enter output string: ")
prob_str = "quad_" + string


def calculate_posterior(theta_c, N):
    """
    return the posterior parameters
    prior is assumed gamma(2,0)
    the true distribution is exponential with theta_c
    M is the input data size
    """
    seed = np.random.random(N)
    log = np.log(seed)
    data = (-1 / theta_c) * log
    a = 2 + N
    b = np.sum(data)
    theta_hat = 1 / np.average(data)
    return a, b, theta_hat, data


def quad(m, theta, x):
    xi = (-1/theta) * np.log(np.random.random(m))
    c = x - xi
    val = np.inner(c, c) / m
    der = 2 * np.average(c)
    return val, der


# def quad_vectorized(n, m, theta, x):
#     xi = np.multiply((-1/theta), np.log(np.random.random((m, n))))
#     c = x - xi
#     val = np.linalg.norm(c, axis=0) ** 2 / m
#     der = 2 * np.average(c, axis=0)
#     return val, der


def collect_samples_empirical(m, x):
    m = int(m)
    return quad(m, theta_hat, x)


def collect_inner_samples(m, theta, x):
    m = int(m)
    return quad(m, theta, x)


def collect_samples(n, m, x):
    sample_list = np.zeros(n)
    derivative_list = np.zeros(n)
    theta = np.random.gamma(post_a, 1 / post_b, n)
    for i in range(n):
        val, der = collect_inner_samples(m, theta[i], x)
        sample_list[i] = val
        derivative_list[i] = der
    return np.array(sample_list), np.array(derivative_list)


# def collect_samples_v2(n, m, x):
#     theta = np.random.gamma(post_a, 1/post_b, n)
#     val, der = quad_vectorized(n, m, theta, x)
#     return val, der


def calc_der_var(n, m, x, alpha):
    sample_list, derivative_list = collect_samples(n, m, x)

    sort_index = np.argsort(sample_list)
    sorted_list = sample_list[sort_index]
    sorted_der = derivative_list[sort_index]

    return sorted_list[int(n * alpha)], sorted_der[int(n * alpha)]


def calc_der_cvar(n, m, x, alpha):
    sample_list, derivative_list = collect_samples(n, m, x)

    var_alpha = np.sort(sample_list)[int(n * alpha)]

    cvar_list = []
    cvar_der_list = []
    for i in range(n):
        if sample_list[i] >= var_alpha:
            cvar_list.append(sample_list[i])
            cvar_der_list.append(derivative_list[i])

    return np.average(cvar_list), np.average(cvar_der_list, 0)


def linear_budget_var(iter_count, alpha, run, x_0=x0, linear_coef=linear_coef0, eps_num=eps_num0, eps_denom=eps_denom0, n_m_ratio=n_m_ratio0):
    """
    start with x_0 and follow the algorithm from there
    use the iterative algorithm and map the evolution of the objective function value
    budget increases linearly as follows: n = n0+t, m = m0 + t/10 etc. The constants might change
    """
    # val_list = []
    # der_list = []
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
        # val_list.append(val)
        # der_list.append(der)
        now = datetime.datetime.now()
        print("run = " + str(run) + " var_ " + str(alpha) + " t = ", t, " x = ", x_list[t], " val = ", val, " der = ", der, " time: ", now-start)
        if (t+1) % 100 == 0:
            if np.std(x_list[-50:]) < 0.005:
                break
    return np.average(x_list[-50:])  # val_list, der_list


def linear_budget_cvar(iter_count, alpha, run, x_0=x0, linear_coef=linear_coef0, eps_num=eps_num0, eps_denom=eps_denom0, n_m_ratio=n_m_ratio0):
    """
    start with x0 and follow the algorithm from there
    use the iterative algorithm and map the evolution of the objective function value
    budget increases linearly as follows: M = n0+t, m = m0 + t/10 etc. The constants might change
    """
    # val_list = []
    # der_list = []
    x_list = [x_0]
    for t in range(iter_count):
        eps = eps_num / (eps_denom + t) ** eps_power
        n = n0 + int(linear_coef * t)
        val, der = calc_der_cvar(n, int(n * n_m_ratio), x_list[t], alpha)
        x_next = max(np.array(x_list[t]) - eps * np.array(der), x_low)  # make sure x is not out of bounds
        x_list.append(x_next)
        # val_list.append(val)
        # der_list.append(der)
        now = datetime.datetime.now()
        print("run = " + str(run) + " cvar_" + str(alpha) + " t = ", t, " x = ", x_list[t], " val = ", val, " der = ", der, " time: ", now-start)
        if (t + 1) % 100 == 0:
            if np.std(x_list[-50:]) < 0.005:
                break
    return np.average(x_list[-50:])  # val_list, der_list


def linear_budget_empirical(iter_count, run, x_0=x0, linear_coef=linear_coef0, eps_num=eps_num0, eps_denom=eps_denom0, n_m_ratio=n_m_ratio0):
    """
    start with x_0 and follow the algorithm from there
    use the iterative algorithm and map the evolution of the objective function value
    budget increases linearly as follows: M = n0+t, m = m0 + t/10 etc. The constants might change
    """
    # val_list = []
    # der_list = []
    x_list = [x_0]
    for t in range(iter_count):
        eps = eps_num / (eps_denom + t) ** eps_power
        n = n0 + int(linear_coef * t)
        val, der = collect_samples_empirical(int(n * n_m_ratio), x_list[t])
        x_next = max(np.array(x_list[t]) - eps * np.array(der), x_low)  # make sure x is not out of bounds
        x_list.append(x_next)
        # val_list.append(val)
        # der_list.append(der)
        now = datetime.datetime.now()
        print("run = " + str(run) + " empirical t = ", t, " x = ", x_list[t], " val = ", val, " der = ", der, " time: ", now-start)
        if (t+1) % 100 == 0:
            if np.std(x_list[-50:]) < 0.005:
                break
    return np.average(x_list[-50:])  # val_list, der_list


if __name__ == "__main__":
    theta_c = 1  # float(input("enter theta_c: "))
    N = int(input("enter input size N: "))
    budget = 5000  # int(input("enter number of iterations: "))
    runs = int(input("enter number of runs: "))
    output = {
        "post_a_b": [],
        "theta_hat": [],
        "data": [],
        "var_0.5": [],
        "var_0.7": [],
        "var_0.9": [],
        "cvar_0.5": [],
        "cvar_0.7": [],
        "cvar_0.9": [],
        "empirical": []
        }
    for i in range(1, runs+1):
        post_a, post_b, theta_hat, data = calculate_posterior(theta_c, N)
        output["post_a_b"].append((post_a, post_b))
        output["theta_hat"].append(theta_hat)
        output["data"].append(data)
        output["var_0.5"].append(linear_budget_var(budget, 0.5, i))
        output["var_0.7"].append(linear_budget_var(budget, 0.7, i))
        output["var_0.9"].append(linear_budget_var(budget, 0.9, i))
        output["cvar_0.5"].append(linear_budget_cvar(budget, 0.5, i))
        output["cvar_0.7"].append(linear_budget_cvar(budget, 0.7, i))
        output["cvar_0.9"].append(linear_budget_cvar(budget, 0.9, i))
        output["empirical"].append(linear_budget_empirical(budget, i))
        np.save("output/quad_combined_" + prob_str + "_N_" + str(N) + "_runs_" + str(runs) + "_output.npy", output)

end = datetime.datetime.now()
print("time: ", end-start)
