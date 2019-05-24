"""
this code aims to show the convergence of the BRO-CVaR problem to the normal distribution
We sample using a fixed x and different posterior distributions for the theta
The aim is to show the convergence of the samples by drawing etc.
"""
import numpy as np
import datetime
from multiprocessing import Pool as ThreadPool
from code_v1.mm1_toy import queue

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


def collect_samples(n, m, x, alpha, beta):
    sample_list = []
    derivative_list = []
    arg_list = []

    for i in range(n):
        theta = np.random.gamma(alpha, 1/beta)
        arg_list.append((m, theta, x))
    pool = ThreadPool()
    results = pool.starmap(collect_inner_samples, arg_list)
    pool.close()
    pool.join()
    for res in results:
        sample_list.append(res[0])
        derivative_list.append(res[1])
    return sample_list, derivative_list


def calculate_posterior(prior_alpha, prior_beta, true_theta, sample_count):
    """
    take the prior, true distribution and sample count
    sample from the true distribution
    calculate the posterior distribution
    """
    arrival_seed = np.random.random(sample_count)
    arrival_log = np.log(arrival_seed)
    arrival = -(1 / true_theta) * arrival_log
    post_alpha = prior_alpha + sample_count
    post_beta = prior_beta + np.sum(arrival)
    return post_alpha, post_beta


def single_run(prior_alpha, prior_beta, true_theta, sample_count, n, m, x, q_alpha):
    """
    handle the single run stuff for a given sample count
    run the calculate_posterior and use the result to collect_samples
    use those samples to calculate the CVaR value
    """
    post_alpha, post_beta = calculate_posterior(prior_alpha, prior_beta, true_theta, sample_count)
    sample_list, derivative_list = collect_samples(n, m, x, post_alpha, post_beta)
    var_alpha = np.sort(sample_list)[int(np.ceil(n * q_alpha))]
    cvar_list = []
    for i in range(n):
        if sample_list[i] >= var_alpha:
            cvar_list.append(sample_list[i])
    return np.average(cvar_list)


def main_run(prior_alpha=2.0, prior_beta=0.0, true_theta=10.0, n=1000, m=1000, x=6.0, q_alpha=0.8, replication=400, budget_list=[10, 100, 1000, 10000]):
    """
    loop through the single run with increasing sample sizes to show the convergence
    save the data and plot it in a meaningful way
    """
    params = {"prior_alpha": prior_alpha, "prior_beta": prior_beta, "true_theta": true_theta,
              "M": n, "m": m, "x": x, "q_alpha": q_alpha, "replication": replication}
    results = {}
    for sample_count in budget_list:
        runs = []
        for i in range(replication):
            print("sample_count: ", sample_count, ", run count: ", i, " time: ", (datetime.datetime.now() - start))
            runs.append(single_run(prior_alpha, prior_beta, true_theta, sample_count, n, m, x, q_alpha))
        results[sample_count] = runs
    output = {"params": params, "results": results}
    np.save("bro_convergence_cvar_alpha_" + str(q_alpha) + "_budget_" + str(budget_list) + ".npy", output)
    return output


if __name__ == "__main__":
    main_run()
