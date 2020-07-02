import datetime
from old.code_v1.sa_params import *
from old.code_v1.prod_inv import prod


start = datetime.datetime.now()

string = input("enter output string: ")
prob = prod
prob_str = "prod_empirical_" + string


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


def linear_budget_empirical(iter_count, x_0=x0, linear_coef=linear_coef0, eps_num=eps_num0, eps_denom=eps_denom0, n_m_ratio=n_m_ratio0):
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
        print("t = ", t, " x = ", x_list[t], " val = ", val, " der = ", der, " time: ", now-begin)
        if (t+1) % 100 == 0:
            np.save("output/" + prob_str + "_empirical_linear" + str(linear_coef) + "_n-m" + str(n_m_ratio) + "_t0=" + str(x_0) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_" + str(eps_power) + "_x", x_list)
            np.save("output/" + prob_str + "_empirical_linear" + str(linear_coef) + "_n-m" + str(n_m_ratio) + "_t0=" + str(x_0) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_" + str(eps_power) + "_val", val_list)
            np.save("output/" + prob_str + "_empirical_linear" + str(linear_coef) + "_n-m" + str(n_m_ratio) + "_t0=" + str(x_0) + "_eps" + str(eps_num) + "-" + str(eps_denom) + "_" + str(eps_power) + "_der", der_list)
    return x_list, val_list, der_list


if __name__ == "__main__":
    theta_hat = float(input("enter theta_hat: "))
    budget = int(input("enter number of iterations: "))
    linear_budget_empirical(budget)

end = datetime.datetime.now()
print("time: ", end-start)
