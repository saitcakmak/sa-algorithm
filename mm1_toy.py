"""
this problem returns the total system time + cost of server for 100 customers in a m/m/1 queue
and the derivative with respect to mu
interarrival (lam) and service time (mu) parameters are given by the caller
cost of server = mu^2 * 10
the formula for derivative is explained in handwritten notes
"""

import numpy as np


M = 100
a = 0.005


def mm1(lam, mu, seed=0):
    if seed:
        np.random.seed(seed)
    arrival_seed = np.random.random(M)
    service_seed = np.random.random(M)
    arrival_log = np.log(arrival_seed)
    service_log = np.log(service_seed)

    arrival = []
    departure = []
    busy_period = []

    arrival.append(-(1/lam) * arrival_log[0])
    departure.append(arrival[0] - (1/mu) * service_log[0])
    busy = service_log[0]
    busy_period.append(busy)
    for i in range(1, M):
        arrival.append(arrival[i-1] - (1/lam) * arrival_log[i])
        if arrival[i] < departure[i-1]:
            busy += service_log[i]
        else:
            busy = service_log[i]
        departure.append(max(arrival[i], departure[i-1]) - (1/mu) * service_log[i])
        busy_period.append(busy)

    system_time = np.asarray(departure) - np.asarray(arrival)
    cost = np.average(system_time) + mu ** 2 * a
    derivative = np.average(busy_period) / mu ** 2 + a * 2 * mu
    return cost, derivative


def queue_with_theta_der(lam, mu, seed=0):
    if seed:
        np.random.seed(seed)
    arrival_seed = np.random.random(M)
    service_seed = np.random.random(M)
    arrival_log = np.log(arrival_seed)
    service_log = np.log(service_seed)

    arrival = []
    departure = []
    busy_period = []

    arrival.append(-(1/lam) * arrival_log[0])
    departure.append(arrival[0] - (1/mu) * service_log[0])
    busy = 0
    last = 0
    busy_period.append(busy)
    for i in range(1, M):
        arrival.append(arrival[i-1] - (1/lam) * arrival_log[i])
        if arrival[i] < departure[i-1]:
            busy = last
        else:
            busy = 0
            last = arrival_log[i]
        departure.append(max(arrival[i], departure[i-1]) - (1/mu) * service_log[i])
        busy_period.append(busy)

    system_time = np.asarray(departure) - np.asarray(arrival)
    cost = np.average(system_time) + mu ** 2 * a
    derivative = -np.average(busy_period) / lam ** 2
    return cost, derivative

