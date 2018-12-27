"""
We first generate the demand and supply numbers from exponential functions.
Then, we take their minimum as the number of rides in the system.
Once we get the number of rides, we calculate the trip times for each ride.
The trip times will be distributed with gamma(k, theta).
For now, let's forget about the optimal price and wage and go with some fixed values.
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
theta = 3

price = 77.3484
wage = 0.8 * price

replication = 1000000


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


def get_ride_times(n):
    """returns the array of ride times for given price and wage"""
    global k, theta
    return np.random.gamma(k, theta, n)


n = number_of_rides(price, wage)
ride_price = price * k * theta
estimated_wage = wage * k * theta
estimated_profit = (ride_price - estimated_wage) * n


def generate_replications(count, num_rides, fare):
    """this generates #count replications and returns the array of realized profits"""
    profits = []
    for i in range(count):
        ride_times = get_ride_times(num_rides)
        real_wage = wage * ride_times
        profit = sum(fare - real_wage)
        profits.append(profit)
    return np.array(profits)


# print(ride_price)
# print(estimated_wage)
# print(real_wage)
#
# print(estimated_profit)
# print(real_profit)

replicated_profits = generate_replications(replication, n, ride_price)

# print(replicated_profits)

avg_profit = np.average(replicated_profits)

plt.hist(replicated_profits, 'auto', label="profits", color="blue")

plt.axvline(estimated_profit, label="estimated_profit = %.2f" % estimated_profit, color="red")


plt.axvline(avg_profit, label="average profit = %.2f" % avg_profit, color="yellow")

plt.xlabel("Profit")
plt.ylabel("Count")

plt.title("Profit distribution of %d replications" % replication)

plt.legend()

print("estimated profit = ", estimated_profit)

print("average profit = ", avg_profit)

lost_profit = sum(estimated_profit - replicated_profits)
print("lost profit = ", lost_profit)

print("ready")

end = datetime.datetime.now()

print(end-start)

plt.show()