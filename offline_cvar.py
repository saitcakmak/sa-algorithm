import datetime
from multiprocessing import Pool as ThreadPool
from sa_params import *
from mm1_toy import mm1


start = datetime.datetime.now()

string = input("enter output string: ")
prob = mm1
prob_str = "mm1_" + string


def calculate_posterior(theta_c, N):
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
    np.save("output/" + "data_" + prob_str + "_theta_" + str(theta_c) + "_N_" + str(N) + ".npy", data)
    a = 2 + N
    b = np.sum(data)
    return a, b


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
    return sample_list, derivative_list


def calc_der(n, m, x):
    sample_list, derivative_list = collect_samples(n, m, x)

    var_alpha = np.sort(sample_list)[int(np.ceil(n * alpha))]

    cvar_list = []
    cvar_der_list = []
    for i in range(n):
        if sample_list[i] >= var_alpha:
            cvar_list.append(sample_list[i])
            cvar_der_list.append(derivative_list[i])

    return np.average(cvar_list, 0), np.average(cvar_der_list, 0)


def linear_budget(iter_count, x_0=x0, linear_coef=linear_coef0, eps_num=eps_num0, eps_denom=eps_denom0, n_m_ratio=n_m_ratio0):
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
        val, der = calc_der(n, int(n * n_m_ratio), x_list[t])
        x_next = max(np.array(x_list[t]) - eps * np.array(der), x_low)  # make sure x is not out of bounds
        x_list.append(x_next)
        val_list.append(val)
        der_list.append(der)
        now = datetime.datetime.now()
        print("t = ", t, " x = ", x_list[t], " val = ", val, " der = ", der, " time: ", now-begin)
        if (t+1) % 100 == 0:
            np.save("output/" + prob_str + "_CVaR" + "_linear" + str(linear_coef) + "_n-m" + str(n_m_ratio) + "_t0=" + str(x_0) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_" + str(eps_power) + "_x", x_list)
            np.save("output/" + prob_str + "_CVaR" + "_linear" + str(linear_coef) + "_n-m" + str(n_m_ratio) + "_t0=" + str(x_0) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_" + str(eps_power) + "_val", val_list)
            np.save("output/" + prob_str + "_CVaR" + "_linear" + str(linear_coef) + "_n-m" + str(n_m_ratio) + "_t0=" + str(x_0) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_" + str(eps_power) + "_der", der_list)
    return x_list, val_list, der_list


if __name__ == "__main__":
    theta_c = int(input("enter theta_c: "))
    N = int(input("enter input size N: "))
    post_a, post_b = calculate_posterior(theta_c, N)
    budget = int(input("enter number of iterations: "))
    linear_budget(budget)

end = datetime.datetime.now()
print("time: ", end-start)