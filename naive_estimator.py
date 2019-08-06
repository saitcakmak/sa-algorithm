import numpy as np
import problem_sampler


def estimator(theta_list, x, m, alpha, rho, prob, seq=0):
    if prob == "simple":
        sampler = problem_sampler.simple_sampler
    elif prob == "two_sided":
        sampler = problem_sampler.two_sided_sampler
    elif prob == "quad":
        sampler = problem_sampler.quad_sampler
    else:
        return -1
    n = len(theta_list)
    samples = np.zeros(n)
    ders = np.zeros(n)
    for i in range(n):
        inner_samples = sampler(theta_list[i], x, m)
        samples[i] = np.average(inner_samples[0])
        ders[i] = np.average(inner_samples[1])

    sort_index = np.argsort(samples)
    samples = samples[sort_index]
    ders = ders[sort_index]

    if rho == "VaR":
        return samples[int(n * alpha)], ders[int(n * alpha)]
    elif rho == "CVaR":
        return np.average(samples[int(n * alpha):]), np.average(ders[int(n * alpha):])
    elif rho == "mean" or "mle":
        return np.average(samples), np.average(ders)
    elif rho == "m_v":
        return np.average(samples) + samples[int(n * alpha)], np.average(ders) + ders[int(n * alpha)]
    elif rho == "m_c":
        return np.average(samples) + np.average(samples[int(n * alpha):]), np.average(ders) + np.average(ders[int(n * alpha):])
