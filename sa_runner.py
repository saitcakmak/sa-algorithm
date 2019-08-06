import numpy as np
from naive_estimator import estimator
import datetime


eps_num = 20
eps_base = 100
x_low = 1


def estimate(x, n, alpha, rho, t_c_list, t_p_list, in_data):
    if rho != "mle":
        index = np.random.randint(100000, size=n)
        t_c = t_c_list[index]
        t_p = t_p_list[index]
        t_list = np.transpose([t_c, t_p])
    else:
        c_rate = 1/np.sum(in_data["cust"])
        p_rate = 1/np.sum(in_data["prov"])
        t_c = np.log(80 / c_rate - 1) / 10
        t_p = - np.log( (1 - p_rate / 20) / (1 + p_rate / 20) ) / 10
        # noinspection PyTypeChecker
        t_list = np.full_like( np.zeros((n, 2)), np.array([t_c, t_p]) )
    m = int(n/10)
    var, der = estimator(t_list, x, m, alpha, rho, "two_sided")
    return var, der


def sa_run(alpha, rho, t_c_list, t_p_list, in_data, out_string="",  x0=5, n0=100, iter_count=1000):
    begin = datetime.datetime.now()
    val_list = []
    der_list = []
    x_list = [x0]
    for t in range(1, iter_count+1):
        eps = eps_num / (eps_base + t) ** 0.8
        n = n0 + int(0.5 * t)
        val, der = estimate(x_list[t-1], n, alpha, rho, t_c_list, t_p_list, in_data)
        x_next = max(np.array(x_list[t-1]) - eps * np.array(der), x_low)  # make sure x is not out of bounds
        x_list.append(x_next)
        val_list.append(val)
        der_list.append(der)
        now = datetime.datetime.now()
        print(rho + "_" + str(alpha) + " t = ", t, " x = ", x_list[t], " val = ", val, " der = ",
              der, " time: ", now - begin)
    np.save("sa_out/" + rho + "_" + str(alpha) + "_" + out_string + "_iter_" + str(iter_count) + "_eps" + str(
        eps_num) + "-" + str(eps_base) + "_x.npy", x_list)
    return x_list, val_list, der_list


def main(alp, rh, text, input_str="1"):
    np.random.seed()
    t_c_list = np.load("mcmc_out/out_c_" + input_str + ".npy")
    t_p_list = np.load("mcmc_out/out_p_" + input_str + ".npy")
    in_data = np.load("input_data/input_data_" + input_str + ".npy").item()
    sa_run(alp, rh, t_c_list, t_p_list, in_data, text)


if __name__ == "__main__":
    alp = float(input("alpha: " ))
    rh = input("rho: ")
    text = input("text: ")

    main(alp, rh, text)
