import numpy as np
import problem_sampler


def estimator(theta_list, x, m, alpha, rho, prob, seq=0):
    """
    Samples from the function and calculates the value of the estimator and the derivative
    :param theta_list: List of theta samples
    :param x: decision x
    :param m: number of inner samples
    :param alpha: risk level
    :param rho: risk measure
    :param prob: the problem to sample from
    :param seq: ignore
    :return: value of the estimator and its derivative
    """
    if prob == "simple":
        sampler = problem_sampler.simple_sampler
    elif prob == "two_sided":
        sampler = problem_sampler.two_sided_sampler
    elif prob == "quad":
        sampler = problem_sampler.quad_sampler
    elif prob == 'normal':
        sampler = problem_sampler.simple_normal_sampler
    else:
        return -1
    if rho == "mean_variance":
        return variance_estimator(theta_list, x, m, alpha, sampler)
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
    elif rho == "mean" or rho == "mle":
        return np.average(samples), np.average(ders)
    elif rho == "m_v":
        return np.average(samples) + samples[int(n * alpha)], np.average(ders) + ders[int(n * alpha)]
    elif rho == "m_c":
        return np.average(samples) + np.average(samples[int(n * alpha):]), np.average(ders) + np.average(ders[int(n * alpha):])


def variance_estimator(theta_list, x, m, alpha, sampler):
    n = len(theta_list)
    samples = np.zeros(n)
    ders = np.zeros(n)
    prod_list = np.zeros(n)
    for i in range(n):
        inner_samples = sampler(theta_list[i], x, m)
        samples[i] = np.average(inner_samples[0])
        ders[i] = np.average(inner_samples[1])
        val_half_1 = np.average(inner_samples[0][:int(m/2)])
        val_half_2 = np.average(inner_samples[0][int(m/2):])
        der_half_1 = np.average(inner_samples[1][:int(m/2)])
        der_half_2 = np.average(inner_samples[1][int(m/2):])
        prod_list[i] = val_half_1 * der_half_2 + val_half_2 * der_half_1

    first_term = np.average(ders)
    second_term = np.average(prod_list)

    third_list = np.zeros(int(n/2))
    for i in range(int(n/2)):
        third_list[i] = samples[i * 2] * ders[i * 2 + 1] + samples[i * 2 + 1] * ders[i * 2]

    third_term = np.average(third_list)

    mean = np.average(samples)
    var = np.std(samples) ** 2
    val = mean + alpha * var
    der = first_term + alpha * (second_term - third_term)
    # print("mean: ", mean, " var: ", var)
    return val, der
