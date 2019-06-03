import numpy as np
import scipy.stats as sci

K_c = 40
K_p = 20
delta = 10 ** -6


def simple_lr(theta, rvs, likelihood, x):
    transpose = np.transpose(rvs)
    l_0 = sci.norm.pdf(transpose[0] - theta[0])
    l_1 = sci.norm.pdf(transpose[1] - theta[1])
    l_trans = np.transpose(likelihood)
    w_0 = l_trans[0] / l_0
    w_1 = l_trans[1] / l_1
    weights = w_0 * w_1
    return weights


def two_sided_lr(theta, rvs, likelihood, x):
    lam = K_c * 2 * np.exp(- theta[0] * x) / (1 + np.exp(- theta[0] * x))
    mu = K_p * (1 - np.exp(- theta[1] * x)) / (1 + np.exp(- theta[1] * x))
    m, dim = np.shape(rvs)
    weights = np.zeros(m)
    for i in range(m):
        rv_ia = rvs[i][0: int(dim/2)]
        rv_is = rvs[i][int(dim/2):]
        likelihood_ia = np.exp(- lam * (rv_ia - delta)) - np.exp(- lam * rv_ia)
        likelihood_is = np.exp(- mu * (rv_is - delta)) - np.exp(- mu * rv_is)
        likelihood_new = np.concatenate((likelihood_ia, likelihood_is))
        weights[i] = np.prod(np.nan_to_num(likelihood_new) / np.nan_to_num(likelihood[i]))
    return weights



