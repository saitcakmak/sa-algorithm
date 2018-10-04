"""
This an implementation of the two-sided queue problem derived in notes
We are trying to minimize an objective as a function of total customer waiting time and price
Total number of customers is fixed - at 100 for now
The arrival of customers follows a distribution - poisson for now - with estimated rate lambda - CVaR variable
The arrival of servers follows a distribution - poisson as well - with rate mu(p)
Waiting time is 0 if there's a server already available, otherwise it's the time until next server arrival.
To keep with the notation, price is theta and lambda is gamma
"""
import numpy as np


N = 100


def two_sided_queue(gamma, theta, seed=0):
    if seed:
        np.random.seed(seed)
    mu = 5 * np.log(theta)  # if this changed, then change w_der calculation as well
    ia_seed = np.random.random(N)
    is_seed = np.random.random(N)
    ia_log = np.log(ia_seed)
    is_log = np.log(is_seed)
    a_log = [ia_log[0]]
    s_log = [is_log[0]]
    for i in range(1, N):
        a_log.append(a_log[i-1] + ia_log[i])
        s_log.append(s_log[i-1] + is_log[i])
    a_val = (-1 / gamma) * np.array(a_log)
    s_val = (-1 / mu) * np.array(s_log)
    w_list = []
    w_der = []
    for i in range(N):
        w_list.append(max(0, s_val[i] - a_val[i]))  # Waiting time for nth customer
        w_der.append(int(w_list[i] > 0) * (1 / mu ** 2) * s_log[i] * 5 / theta)  # derivative of it
    wait = np.sum(w_list)
    obj = theta * wait + theta ** 2
    der = wait + theta * np.sum(w_der) + 2 * theta
    return obj, der
