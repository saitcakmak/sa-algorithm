import datetime
from multiprocessing import Pool as ThreadPool
from theta_params import mu_theta, std_theta
from sa_params import *
from two_sided_queue import *

start = datetime.datetime.now()

prob = two_sided_ext
prob_str = "two_sided_ext"


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
    sample_list = []
    derivative_list = []
    arg_list = []

    for i in range(n):
        theta = np.random.multivariate_normal(mu_theta, std_theta)
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

    return np.average(cvar_list), np.average(cvar_der_list, 0)


def linear_budget(iter_count, x_0=x0, linear_coef=linear_coef0, eps_num=eps_num0, eps_denom=eps_denom0, n_m_ratio=n_m_ratio0):
    """
    start with x0 and follow the algorithm from there
    use the iterative algorithm and map the evolution of the objective function value
    budget increases linearly as follows: n = n0+k, m = m0 + k/10 etc. The constants might change
    :return:
    """
    begin = datetime.datetime.now()
    val_list = []
    der_list = []
    eps_list = []
    x_list = [x_0]
    for k in range(iter_count):
        eps = eps_num / (eps_denom + k) ** eps_power
        n = n0 + int(linear_coef * k)
        val, der = calc_der(n, int(n * n_m_ratio), x_list[k])
        x_next = max(np.array(x_list[k]) - eps * np.array(der), x_low)  # make sure x is not out of bounds
        x_list.append(x_next)
        val_list.append(val)
        der_list.append(der)
        eps_list.append(eps)
        now = datetime.datetime.now()
        print("k = ", k, " x = ", x_list[k], " val = ", val, " der = ", der, " time: ", now-begin)
        if k % 100 == 0:
            np.save(prob_str + "_CVaR" + "_linear" + str(linear_coef) + "_n-m" + str(n_m_ratio) + "_t0=" + str(x_0) + "_mu" + str(mu_theta) + "_std" + str(std_theta) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_" + str(eps_power) + "_iter_" + str(k) + "_x", x_list)
            np.save(prob_str + "_CVaR" + "_linear" + str(linear_coef) + "_n-m" + str(n_m_ratio) + "_t0=" + str(x_0) + "_mu" + str(mu_theta) + "_std" + str(std_theta) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_" + str(eps_power) + "_iter_" + str(k) + "_val", val_list)
            np.save(prob_str + "_CVaR" + "_linear" + str(linear_coef) + "_n-m" + str(n_m_ratio) + "_t0=" + str(x_0) + "_mu" + str(mu_theta) + "_std" + str(std_theta) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_" + str(eps_power) + "_iter_" + str(k) + "_der", der_list)
    return x_list, val_list, der_list, eps_list


if __name__ == "__main__":
    linear_budget(1001)

end = datetime.datetime.now()
print("time: ", end-start)
