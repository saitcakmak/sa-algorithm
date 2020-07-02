"""
this code aims to show the convergence of the BRO-CVaR problem to the normal distribution
We sample using a fixed x and different posterior distributions for the theta
The aim is to show the convergence of the samples by drawing etc
"""
import numpy as np
import datetime
from multiprocessing import Pool as ThreadPool
from old.code_v1.mm1_toy import queue

start = datetime.datetime.now()

prob = queue
prob_str = "queue"


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


def collect_samples(n, m, x, theta):
    sample_list = []
    derivative_list = []
    arg_list = []

    for i in range(n):
        arg_list.append((m, theta, x))
    pool = ThreadPool()
    results = pool.starmap(collect_inner_samples, arg_list)
    pool.close()
    pool.join()
    for res in results:
        sample_list.append(res[0])
        derivative_list.append(res[1])
    return sample_list, derivative_list


def single_run(true_theta, n, m, x, q_alpha):
    """
    handle the single run stuff for a given sample count
    run the calculate_posterior and use the result to collect_samples
    use those samples to calculate the CVaR value
    """
    sample_list, derivative_list = collect_samples(n, m, x, true_theta)
    var_alpha = np.sort(sample_list)[int(np.ceil(n * q_alpha))]
    cvar_list = []
    for i in range(n):
        if sample_list[i] >= var_alpha:
            cvar_list.append(sample_list[i])
    return np.average(cvar_list)


def main_run(true_theta=10.0, n=1000, m=1000, x=6.0, q_alpha=0.8, replication=100):
    """
    loop through the single run with increasing sample sizes to show the convergence
    save the data and plot it in a meaningful way
    """
    params = {"true_theta": true_theta,
              "M": n, "m": m, "x": x, "q_alpha": q_alpha, "replication": replication}
    results = {}
    runs = []
    for i in range(replication):
        print("run count: ", i, " time: ", (datetime.datetime.now() - start))
        runs.append(single_run(true_theta, n, m, x, q_alpha))
    results["true"] = runs
    output = {"params": params, "results": results}
    np.save("bro_error_cvar.npy", output)
    return output


if __name__ == "__main__":
    print(main_run())
