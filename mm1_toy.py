"""
this problem returns the total system time + cost of server for 100 customers in a m/m/1 queue
and the derivative with respect to mu
interarrival (lam) and service time (mu) parameters are given by the caller
cost of server = mu^2 * 10
the formula for derivative is explained in handwritten notes
"""

import numpy as np


N = 100


def queue(lam, mu, seed=0):
    if seed:
        np.random.seed(seed)
    arrival_seed = np.random.random(N)
    service_seed = np.random.random(N)
    arrival_log = np.log(arrival_seed)
    service_log = np.log(service_seed)

    arrival = []
    departure = []
    busy_period = []

    arrival.append(-(1/lam) * arrival_log[0])
    departure.append(arrival[0] - (1/mu) * service_log[0])
    busy = service_log[0]
    busy_period.append(busy)
    for i in range(1, N):
        arrival.append(arrival[i-1] - (1/lam) * arrival_log[i])
        if arrival[i] < departure[i-1]:
            busy += service_log[i]
        else:
            busy = service_log[i]
        departure.append(max(arrival[i], departure[i-1]) - (1/mu) * service_log[i])
        busy_period.append(busy)

    system_time = np.asarray(departure) - np.asarray(arrival)
    cost = np.sum(system_time) + mu ** 2 * 10
    derivative = sum(busy_period) / mu ** 2 + 20 * mu
    return cost, derivative


def queue_with_theta_der(lam, mu, seed=0):
    if seed:
        np.random.seed(seed)
    arrival_seed = np.random.random(N)
    service_seed = np.random.random(N)
    arrival_log = np.log(arrival_seed)
    service_log = np.log(service_seed)

    arrival = []
    departure = []
    busy_period = []

    arrival.append(-(1/lam) * arrival_log[0])
    departure.append(arrival[0] - (1/mu) * service_log[0])
    busy = 0
    busy_period.append(busy)
    for i in range(1, N):
        arrival.append(arrival[i-1] - (1/lam) * arrival_log[i])
        if arrival[i] < departure[i-1]:
            busy += arrival_log[i]
        else:
            busy = 0
        departure.append(max(arrival[i], departure[i-1]) - (1/mu) * service_log[i])
        busy_period.append(busy)

    system_time = np.asarray(departure) - np.asarray(arrival)
    cost = np.sum(system_time)
    derivative = sum(busy_period) / mu ** 2
    return cost, derivative

