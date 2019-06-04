import numpy as np
from naive_estimator import estimator
import datetime


t_c_list = np.zeros((5, 100000))
t_p_list = np.zeros((5, 100000))

t_c_list[0] = np.load("mcmc_out/out_c_try.npy")
t_p_list[0] = np.load("mcmc_out/out_p_try.npy")
t_c_list[1] = np.load("mcmc_out/out_c_try2.npy")
t_p_list[1] = np.load("mcmc_out/out_p_try2.npy")
t_c_list[2] = np.load("mcmc_out/out_c_try3.npy")
t_p_list[2] = np.load("mcmc_out/out_p_try3.npy")
t_c_list[3] = np.load("mcmc_out/out_c_try4.npy")
t_p_list[3] = np.load("mcmc_out/out_p_try4.npy")
t_c_list[4] = np.load("mcmc_out/out_c_try5.npy")
t_p_list[4] = np.load("mcmc_out/out_p_try5.npy")

eps_num = 20
eps_base = 100
x_low = 1


def estimate(x, n, alpha, rho, block):
    index = np.random.randint(100000, size=n)
    t_c = t_c_list[block][index]
    t_p = t_p_list[block][index]
    t_list = np.transpose([t_c, t_p])
    m = int(n/10)
    var, der = estimator(t_list, x, m, alpha, rho, "two_sided")
    return var, der


def online_run(alpha, rho, out_string="", x0=5, n0=300, iter_count=1000):
    begin = datetime.datetime.now()
    val_list = []
    der_list = []
    x_list = [x0]
    for t in range(1, iter_count+1):
        eps = eps_num / (eps_base + ((t-1) % 200) + 1) ** 0.8
        n = n0 + int(0.5 * ((t-1) % 200 + 1))
        val, der = estimate(x_list[t-1], n, alpha, rho, int((t-1)/200))
        x_next = max(np.array(x_list[t-1]) - eps * np.array(der), x_low)  # make sure x is not out of bounds
        x_list.append(x_next)
        val_list.append(val)
        der_list.append(der)
        now = datetime.datetime.now()
        print(rho + "_" + str(alpha) + " t = ", t, " x = ", x_list[t], " val = ", val, " der = ",
              der, " time: ", now - begin)
        if t % 100 == 0:
            np.save("sa_out/online_" + rho + "_" + str(alpha) + "_" + out_string + "_iter_" + str(iter_count) + "_eps"
                    + str(eps_num) + "-" + str(eps_base) + "_x.npy", x_list)
            np.save("sa_out/online_" + rho + "_" + str(alpha) + "_" + out_string + "_iter_" + str(iter_count) + "_eps"
                    + str(eps_num) + "-" + str(eps_base) + "_val.npy", val_list)
            np.save("sa_out/online_" + rho + "_" + str(alpha) + "_" + out_string + "_iter_" + str(iter_count) + "_eps"
                    + str(eps_num) + "-" + str(eps_base) + "_der.npy", der_list)
    return x_list, val_list, der_list