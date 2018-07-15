"""
We first generate the demand and supply numbers from exponential functions.
Then, we take their minimum as the number of rides in the system.
Once we get the number of rides, we calculate the trip times for each ride.
The trip times will be distributed with gamma(k, theta).
For now, let's forget about the optimal price and wage and go with some fixed values.

In this case, different from comparison_trial_1, we will have theta normally distributed.
"""
import numpy as np
import matplotlib.pyplot as plt
import datetime

start = datetime.datetime.now()

LAMBDA = 1000
K = 1000
k1 = 0.01
k2 = 0.01
k = 4
mean_theta = 3
std_theta = 1

price = 77.3484
wage = 0.8 * price

replication = 100000


def demand(p):
    """returns the demand value for the given price"""
    global LAMBDA, k1
    return LAMBDA * (np.exp(-k1 * p))


def supply(w):
    """returns the driver supply for the given wage"""
    global K, k2
    return K * (1 - np.exp(-k2 * w))


def number_of_rides(p, w):
    """returns the number of rides for the given price and wage"""
    return int(min(demand(p), supply(w)))


def get_theta_ride_times(n):
    """returns the array of theta and ride times for given price and wage"""
    global k, mean_theta, std_theta
    theta = np.maximum(np.random.normal(mean_theta, std_theta, n), 0)
    return theta, np.random.gamma(k, theta, n)


def generate_replications(count, num_rides):
    """this generates #count replications and returns the array of realized profits"""
    global price, wage
    profits = []
    expected_profits = []
    for i in range(count):
        theta, ride_times = get_theta_ride_times(num_rides)
        fare = k * theta * price
        expected_wage = k * theta * wage
        real_wage = wage * ride_times
        profit = sum(fare - real_wage)
        expected_profit = sum(fare - expected_wage)
        expected_profits.append(expected_profit)
        profits.append(profit)
    return np.array(expected_profits), np.array(profits)


n = number_of_rides(price, wage)

expected_replicated_profits, replicated_profits = generate_replications(replication, n)

estimated_profit = np.average(expected_replicated_profits)

avg_profit = np.average(replicated_profits)

plt.hist(replicated_profits, 'auto', label="profits", color="blue")

plt.axvline(estimated_profit, label="estimated_profit = %.2f" % estimated_profit, color="red")

plt.axvline(avg_profit, label="average profit = %.2f" % avg_profit, color="yellow")

plt.xlabel("Profit")
plt.ylabel("Count")

plt.title("Profit distribution of %d replications with Norm(%d,%d) theta" % (replication, mean_theta, std_theta))

plt.legend()

print("estimated profit = ", estimated_profit)

print("average profit = ", avg_profit)

lost_profit = sum(estimated_profit - replicated_profits)
print("lost profit = ", lost_profit)

print("ready")

end = datetime.datetime.now()

print(end-start)

plt.show()