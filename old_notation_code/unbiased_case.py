"""
in this example we simulate without any input uncertainty and with unbiased estimator of distance and time
P_c = 10 * MU_d + 5 * MU_t
P_f = 10 * d + 5 * t
C(P_c) = 1 - 0.005 P_c (this will be refined later)
"""
import numpy as np
import matplotlib.pyplot as plt

MU_d = 5
SIGMA_d = 1

MU_t = 5
SIGMA_t = 2


def single_ride():
    global MU_d
    global MU_t
    global SIGMA_d
    global SIGMA_t
    P_c = 10 * MU_d + 5 * MU_t
    d = np.random.normal(MU_d, SIGMA_d)
    t = np.random.normal(MU_t, SIGMA_t)
    P_f = 10 * d + 5 * t
    exp_profit = P_c - P_f
    conv_prob = 1 - 0.005 * P_c
    print(conv_prob)
    if np.random.random_sample() <= conv_prob:
        return exp_profit


run_results = []

for i in range(100000):
    result = single_ride()
    if result is not None:
        run_results.append(result)

print(run_results)
print(len(run_results))
print('mean: %f' %(np.mean(run_results)))

num_bins = 50

fig, ax = plt.subplots()

n, bins, patches = ax.hist(run_results, num_bins, normed=1)

ax.set_xlabel('profit')
ax.set_ylabel('count')
ax.set_title('Profit Distribution Under Unbiased Pricing')

plt.show()


