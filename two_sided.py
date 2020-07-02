"""
This an implementation of the two-sided queue problem derived in notes
We are trying to minimize an objective as a function of total customer waiting time and price
Total number of customers is fixed - at 100 for now
The arrival of customers follows a distribution - poisson for now - with estimated rate lambda - CVaR variable
The arrival of servers follows a distribution - poisson as well - with rate mu(p)
Waiting time is 0 if there's a server already available, otherwise it's the time until next server arrival.
To keep with the notation, price is x and lambda is theta
"""
import numpy as np


M = 100
a = 1/25
K_c = 40
K_p = 20
delta = 10 ** -6


def two_sided(theta, x):
    """
    Two-sided problem objective
    :param theta:
    :param x:
    :return:
    """
    lam = K_c * 2 * np.exp(- theta[0] * x) / (1 + np.exp(- theta[0] * x))
    # if this is changed, then change derivative calculation as well
    lam_prime = - K_c * 2 * theta[0] * np.exp(- theta[0] * x) / (1 + np.exp(- theta[0] * x)) ** 2
    mu = K_p * (1 - np.exp(- theta[1] * x)) / (1 + np.exp(- theta[1] * x))  # same with this
    mu_prime = K_p * 2 * theta[1] * np.exp(- theta[1] * x) / (1 + np.exp(- theta[1] * x)) ** 2
    ia_seed = np.random.random(M)
    is_seed = np.random.random(M)
    ia_log = np.log(ia_seed)
    is_log = np.log(is_seed)
    a_log = np.zeros(M)
    s_log = np.zeros(M)
    a_log[0] = ia_log[0]
    s_log[0] = is_log[0]
    for i in range(1, M):
        a_log[i] = a_log[i-1] + ia_log[i]
        s_log[i] = s_log[i-1] + is_log[i]
    a_val = (-1 / lam) * np.array(a_log)
    s_val = (-1 / mu) * np.array(s_log)
    w_list = np.maximum(0, s_val - a_val)  # waiting times
    w_der = (w_list > 0) * ((1 / mu ** 2) * s_log * mu_prime - (1 / lam ** 2) * a_log * lam_prime)  # the derivatives
    wait = np.average(w_list)
    wait_der = np.average(w_der)
    obj = wait - a * x * lam  # update both
    der = wait_der - a * (lam + x * lam_prime)
    return obj, der


def two_sided_lr(theta, x):
    """
    Two sided objective that returns likelihood for IS or LR calculations.
    :param theta:
    :param x:
    :return:
    """
    lam = K_c * 2 * np.exp(- theta[0] * x) / (1 + np.exp(- theta[0] * x))
    # if this is changed, then change derivative calculation as well
    lam_prime = - K_c * 2 * theta[0] * np.exp(- theta[0] * x) / (1 + np.exp(- theta[0] * x)) ** 2
    mu = K_p * (1 - np.exp(- theta[1] * x)) / (1 + np.exp(- theta[1] * x))  # same with this
    mu_prime = K_p * 2 * theta[1] * np.exp(- theta[1] * x) / (1 + np.exp(- theta[1] * x)) ** 2
    ia_seed = np.random.random(M)
    is_seed = np.random.random(M)
    ia_log = np.log(ia_seed)
    is_log = np.log(is_seed)
    a_log = np.zeros(M)
    s_log = np.zeros(M)
    a_log[0] = ia_log[0]
    s_log[0] = is_log[0]
    for i in range(1, M):
        a_log[i] = a_log[i-1] + ia_log[i]
        s_log[i] = s_log[i-1] + is_log[i]
    a_val = (-1 / lam) * np.array(a_log)
    s_val = (-1 / mu) * np.array(s_log)
    w_list = np.maximum(0, s_val - a_val)  # waiting times
    w_der = (w_list > 0) * ((1 / mu ** 2) * s_log * mu_prime - (1 / lam ** 2) * a_log * lam_prime)  # the derivatives
    wait = np.average(w_list)
    wait_der = np.average(w_der)
    obj = wait - a * x * lam  # update both
    der = wait_der - a * (lam + x * lam_prime)
    rv_ia = (-1 / lam) * ia_log
    rv_is = (-1 / mu) * is_log
    rvs = np.concatenate((rv_ia, rv_is))
    likelihood_ia = np.exp(- lam * (rv_ia - delta)) - np.exp(- lam * rv_ia)
    likelihood_is = np.exp(- mu * (rv_is - delta)) - np.exp(- mu * rv_is)
    likelihood = np.concatenate((likelihood_ia, likelihood_is))
    return obj, der, rvs, likelihood
