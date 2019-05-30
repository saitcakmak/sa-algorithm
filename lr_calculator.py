import numpy as np
import scipy.stats as sci


def simple_lr(theta, rvs, likelihoods):
    """
    this takes into account both xi's
    """
    transpose = np.transpose(rvs)
    l_0 = sci.norm.pdf(transpose[0] - theta[0])
    l_1 = sci.norm.pdf(transpose[1] - theta[1])
    l_trans = np.transpose(likelihoods)
    w_0 = l_trans[0] / l_0
    w_1 = l_trans[1] / l_1
    weights = w_0 * w_1
    return weights


def simple_lr_v2(theta, rvs, likelihoods):
    """
    only consider xi_1 since that's what is relevant for the derivative.
    """
    transpose = np.transpose(rvs)
    l_0 = sci.norm.pdf(transpose[0] - theta[0])
    l_trans = np.transpose(likelihoods)
    w_0 = l_trans[0] / l_0
    weights = w_0
    return weights