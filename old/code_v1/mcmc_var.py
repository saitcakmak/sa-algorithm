import datetime
from multiprocessing import Pool as ThreadPool
from old.code_v1.sa_params import *
from old.code_v1.two_sided_queue import *


start = datetime.datetime.now()

string = input("enter output string: ")
prob = two_sided_ext
prob_str = "two_sided_ext_" + string
data_file = "output/" + input("enter data file name complete with extension: ")
theta_samples = np.load(data_file).tolist()


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
    global theta_samples
    sample_list = []
    derivative_list = []
    arg_list = []
    for i in range(n):
        index = int(np.random.rand()*100000)
        theta = theta_samples[index]
        arg_list.append((m, theta, x))
    pool = ThreadPool()
    results = pool.starmap(collect_inner_samples, arg_list)
    pool.close()
    pool.join()
    for res in results:
        sample_list.append(res[0])
        derivative_list.append(res[1])
    return np.array(sample_list), np.array(derivative_list)


def calc_der(n, m, x):
    sample_list, derivative_list = collect_samples(n, m, x)

    sort_index = np.argsort(sample_list)
    sorted_list = sample_list[sort_index]
    sorted_der = derivative_list[sort_index]

    return sorted_list[int(np.ceil(n * alpha))], sorted_der[int(np.ceil(n * alpha))]


def linear_budget(iter_count, x_0=x0, linear_coef=linear_coef0, eps_num=eps_num0, eps_denom=eps_denom0, n_m_ratio=n_m_ratio0):
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
        k = int(n / 100)  # k in the algorithm definition
        v_list = []
        d_list = []
        for i in range(k):
            v, d = calc_der(n, int(n * n_m_ratio), x_list[t])
            v_list.append(v)
            d_list.append(d)
        val = np.average(v_list)
        der = np.average(d_list)
        x_next = max(np.array(x_list[t]) - eps * np.array(der), x_low)  # make sure x is not out of bounds
        x_list.append(x_next)
        val_list.append(val)
        der_list.append(der)
        now = datetime.datetime.now()
        print("t = ", t, " x = ", x_list[t], " val = ", val, " der = ", der, " time: ", now-begin)
        if (t+1) % 100 == 0:
            np.save("output/" + prob_str + "_VaR" + "_linear" + str(linear_coef) + "_n-m" + str(n_m_ratio) + "_t0=" + str(x_0) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_" + str(eps_power) + "_x", x_list)
            np.save("output/" + prob_str + "_VaR" + "_linear" + str(linear_coef) + "_n-m" + str(n_m_ratio) + "_t0=" + str(x_0) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_" + str(eps_power) + "_val", val_list)
            np.save("output/" + prob_str + "_VaR" + "_linear" + str(linear_coef) + "_n-m" + str(n_m_ratio) + "_t0=" + str(x_0) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_" + str(eps_power) + "_der", der_list)
    return x_list, val_list, der_list


if __name__ == "__main__":
    budget = int(input("enter number of iterations: "))
    linear_budget(budget)

end = datetime.datetime.now()
print("time: ", end-start)
