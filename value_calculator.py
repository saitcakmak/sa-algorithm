import numpy as np
from multiprocessing import Pool
from problem_sampler import two_sided_sampler
import datetime

start = datetime.datetime.now()

t_c_list = np.load("mcmc_out/out_c_1.npy")
t_p_list = np.load("mcmc_out/out_p_1.npy")


def run(theta, x, m):
    np.random.seed()
    vals, ders = two_sided_sampler(theta, x, m)
    return np.average(vals), np.average(ders)


def estimate(rho, alpha, x, n=1000):
    m = int(n/10)
    arg_list = []
    for i in range(n):
        ind_c = np.random.randint(0, 100000)
        ind_p = np.random.randint(0, 100000)
        arg_list.append(((t_c_list[ind_c], t_p_list[ind_p]), x, m))

    pool = Pool()
    results = pool.starmap(run, arg_list, 25)
    pool.close()
    pool.join()

    end = datetime.datetime.now()
    # print("done, ", end-start)
    results = np.array(results)
    vals = results[:, 0]
    vals = np.sort(vals)
    if rho == "CVaR":
        return np.average(vals[int(n * alpha):])
    elif rho == "VaR":
        return vals[int(n * alpha)]
    else:
        return 0


def big_run(n=1000, seed=0):
    for i in range(50):
        np.random.seed(seed)
        x = 23 + 0.02 * i
        val = estimate("CVaR", 0.5, x, n)
        print(x, val)
