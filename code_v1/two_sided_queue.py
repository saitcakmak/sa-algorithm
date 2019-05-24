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


def two_sided_ext(theta, x, seed=0):
    if seed:
        np.random.seed(seed)
    lam = theta[0] * np.exp(- theta[1] * x) / (1 + np.exp(- theta[1] * x))  # if this is changed, then change derivative calculation as well
    lam_prime = - theta[0] * theta[1] * np.exp(- theta[1] * x) / (1 + np.exp(- theta[1] * x)) ** 2
    mu = theta[2] / (1 + np.exp(- theta[3] * x))  # same with this
    mu_prime = theta[2] * theta[3] * np.exp(- theta[3] * x) / (1 + np.exp(- theta[3] * x)) ** 2
    ia_seed = np.random.random(M)
    is_seed = np.random.random(M)
    ia_log = np.log(ia_seed)
    is_log = np.log(is_seed)
    a_log = [ia_log[0]]
    s_log = [is_log[0]]
    for i in range(1, M):
        a_log.append(a_log[i-1] + ia_log[i])
        s_log.append(s_log[i-1] + is_log[i])
    a_val = (-1 / lam) * np.array(a_log)
    s_val = (-1 / mu) * np.array(s_log)
    w_list = []
    w_der = []
    for i in range(M):
        w_list.append(max(0, s_val[i] - a_val[i]))  # Waiting time for nth customer
        w_der.append(int(w_list[i] > 0) * ((1 / mu ** 2) * s_log[i] * mu_prime - (1 / lam ** 2) * a_log[i] * lam_prime))  # derivative of it
    wait = np.average(w_list)
    wait_der = np.average(w_der)
    obj = wait - a * x * lam  # update both
    der = wait_der - a * (lam + x * lam_prime)
    return obj, der
