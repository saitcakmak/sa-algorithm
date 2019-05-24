import numpy as np
import problem_sampler


def estimator(theta_list, x, m, alpha, rho):
    n = len(theta_list)
    samples = np.zeros(n)
    ders = np.zeros(n)
    for i in range(n):
        inner_samples = problem_sampler.sampler(theta_list[i], x, m)
        samples[i] = np.average(inner_samples[0])
        ders[i] = np.average(inner_samples[1])

    sort_index = np.argsort(samples)
    samples = samples[sort_index]
    ders = ders[sort_index]

    if rho == "VaR":
        return samples[int(n * alpha)], ders[int(n * alpha)]
    elif rho == "CVaR":
        return np.average(samples[int(n * alpha): n]), np.average(ders[int(n * alpha): n])
