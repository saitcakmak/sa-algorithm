import datetime
from multiprocessing import Pool as ThreadPool
from sa_params import *
from mm1_toy import mm1


start = datetime.datetime.now()

string = input("enter output string: ")
prob = mm1
prob_str = "mm1_" + string


def collect_samples_empirical(m, x):
    global prob, theta_hat
    np.random.seed()
    inner_list = []
    inner_derivative_list = []
    for j in range(m):
        val, der = prob(theta_hat, x)
        inner_list.append(val)
        inner_derivative_list.append(der)
    return np.average(inner_list), np.average(inner_derivative_list, 0)


def calculate_posterior(theta_c, N, run):
    """
    return the posterior parameters
    prior is assumed gamma(2,0)
    the true distribution is exponential with theta_c
    M is the input data size
    """
    global prob_str
    seed = np.random.random(N)
    log = np.log(seed)
    data = (-1 / theta_c) * log
    # np.save("output/" + "data_" + prob_str + "_theta_" + str(theta_c) + "_N_" + str(N) + "_run_" + str(run) + ".npy", data)
    a = 2 + N
    b = np.sum(data)
    theta_hat = 1 / np.average(data)
    return a, b, theta_hat, data


def collect_inner_samples(m, theta, x):
    global prob
    np.random.seed()
    inner_list = []
    inner_derivative_list = []
    for j in range(m):
        val, der = prob(theta, x)
        inner_list.append(val)
        inner_derivative_list.append(der)
    return np.average(inner_list), np.average(inner_derivative_list, 0)


def collect_samples(n, m, x):
    global post_a, post_b
    sample_list = []
    derivative_list = []
    arg_list = []
    for i in range(n):
        theta = np.random.gamma(post_a, 1/post_b)
        arg_list.append((m, theta, x))
    pool = ThreadPool()
    results = pool.starmap(collect_inner_samples, arg_list)
    pool.close()
    pool.join()
    for res in results:
        sample_list.append(res[0])
        derivative_list.append(res[1])
    return np.array(sample_list), np.array(derivative_list)


def calc_der_var(n, m, x, alpha):
    sample_list, derivative_list = collect_samples(n, m, x)

    sort_index = np.argsort(sample_list)
    sorted_list = sample_list[sort_index]
    sorted_der = derivative_list[sort_index]

    return sorted_list[int(np.ceil(n * alpha))], sorted_der[int(np.ceil(n * alpha))]


def calc_der_cvar(n, m, x, alpha):
    sample_list, derivative_list = collect_samples(n, m, x)

    var_alpha = np.sort(sample_list)[int(np.ceil(n * alpha))]

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
        # if (t+1) % 100 == 0:
        #     np.save("output/" + prob_str + "_run_" + str(run) + "_VaR_" + str(alpha) + "_linear" + str(linear_coef) + "_n-m" + str(n_m_ratio) + "_t0=" + str(x_0) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_" + str(eps_power) + "_x", x_list)
        #     np.save("output/" + prob_str + "_run_" + str(run) + "_VaR_" + str(alpha) + "_linear" + str(linear_coef) + "_n-m" + str(n_m_ratio) + "_t0=" + str(x_0) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_" + str(eps_power) + "_val", val_list)
        #     np.save("output/" + prob_str + "_run_" + str(run) + "_VaR_" + str(alpha) + "_linear" + str(linear_coef) + "_n-m" + str(n_m_ratio) + "_t0=" + str(x_0) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_" + str(eps_power) + "_der", der_list)
    return x_list, val_list, der_list


def linear_budget_cvar(iter_count, alpha, run, x_0=x0, linear_coef=linear_coef0, eps_num=eps_num0, eps_denom=eps_denom0, n_m_ratio=n_m_ratio0):
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
        # if (t+1) % 100 == 0:
        #     np.save("output/" + prob_str + "_run_" + str(run) + "_CVaR_" + str(alpha) + "_linear" + str(linear_coef) + "_n-m" + str(n_m_ratio) + "_t0=" + str(x_0) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_" + str(eps_power) + "_x", x_list)
        #     np.save("output/" + prob_str + "_run_" + str(run) + "_CVaR_" + str(alpha) + "_linear" + str(linear_coef) + "_n-m" + str(n_m_ratio) + "_t0=" + str(x_0) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_" + str(eps_power) + "_val", val_list)
        #     np.save("output/" + prob_str + "_run_" + str(run) + "_CVaR_" + str(alpha) + "_linear" + str(linear_coef) + "_n-m" + str(n_m_ratio) + "_t0=" + str(x_0) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_" + str(eps_power) + "_der", der_list)
    return x_list, val_list, der_list


def linear_budget_empirical(iter_count, run, x_0=x0, linear_coef=linear_coef0, eps_num=eps_num0, eps_denom=eps_denom0, n_m_ratio=n_m_ratio0):
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


if __name__ == "__main__":
    theta_c = int(input("enter theta_c: "))
    N = int(input("enter input size N: "))
    budget = int(input("enter number of iterations: "))
    runs = int(input("enter number of runs: "))
    output = {
        "post_a_b": [],
        "theta_hat": [],
        "data": [],
        "var_0.5": [],
        "var_0.6": [],
        "var_0.7": [],
        "var_0.8": [],
        "var_0.9": [],
        "cvar_0.5": [],
        "cvar_0.6": [],
        "cvar_0.7": [],
        "cvar_0.8": [],
        "cvar_0.9": [],
        "empirical": []
        }
    for i in range(1, runs+1):
        post_a, post_b, theta_hat, data = calculate_posterior(theta_c, N, i)
        output["post_a_b"].append((post_a, post_b))
        output["theta_hat"].append(theta_hat)
        output["data"].append(data)
        output["var_0.5"].append(linear_budget_var(budget, 0.5, i))
        output["var_0.6"].append(linear_budget_var(budget, 0.6, i))
        output["var_0.7"].append(linear_budget_var(budget, 0.7, i))
        output["var_0.8"].append(linear_budget_var(budget, 0.8, i))
        output["var_0.9"].append(linear_budget_var(budget, 0.9, i))
        output["cvar_0.5"].append(linear_budget_cvar(budget, 0.5, i))
        output["cvar_0.6"].append(linear_budget_cvar(budget, 0.6, i))
        output["cvar_0.7"].append(linear_budget_cvar(budget, 0.7, i))
        output["cvar_0.8"].append(linear_budget_cvar(budget, 0.8, i))
        output["cvar_0.9"].append(linear_budget_cvar(budget, 0.9, i))
        output["empirical"].append(linear_budget_empirical(budget, i))
    np.save("output/combined_" + prob_str + "_N_" + str(N) + "_output.npy", output)

end = datetime.datetime.now()
print("time: ", end-start)
