import numpy as np
# TODO: import the problem

prob = pass


def sampler(theta, x, m):
    samples = []
    ders = []

    for j in range(m):
        sample = prob(theta, x)
        samples.append(sample[0])
        ders.append(sample[1])
    return samples, ders


def sampler_lr(theta, x, m):
    samples = []
    ders = []
    rvs = []

    for j in range(m):
        sample = prob(theta, x)
        samples.append(sample[0])
        ders.append(sample[1])
        rvs.append(sample[2])
    return samples, ders, rvs
