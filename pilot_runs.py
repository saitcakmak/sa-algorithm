import numpy as np
import problem_sampler


def stochastic_uncertainty(theta, x, m):
    samples = problem_sampler.two_sided_sampler(theta, x, m)[0]
    std = np.std(samples)
    return std


def input_uncertainty(x, m, n):
    t_c_list = np.load("mcmc_out/out_c_try.npy")
    t_p_list = np.load("mcmc_out/out_p_try.npy")
    index = np.random.randint(100000, size=n)
    t_c = t_c_list[index]
    t_p = t_p_list[index]
    t_list = np.transpose([t_c, t_p])
    samples = np.zeros(n)
    for i in range(n):
        samples[i] = np.average(problem_sampler.two_sided_sampler(t_list[i], x, m)[0])
    std = np.std(samples)
    return std


def overall_uncertainty(x, m, n, k):
    t_c_list = np.load("mcmc_out/out_c_try.npy")
    t_p_list = np.load("mcmc_out/out_p_try.npy")
    reps = np.zeros(k)
    for j in range(k):
        index = np.random.randint(100000, size=n)
        t_c = t_c_list[index]
        t_p = t_p_list[index]
        t_list = np.transpose([t_c, t_p])
        samples = np.zeros(n)
        for i in range(n):
            samples[i] = np.average(problem_sampler.two_sided_sampler(t_list[i], x, m)[0])
        reps[j] = np.average(samples)
    std = np.std(reps)
    return std