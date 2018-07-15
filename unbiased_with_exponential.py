"""
in this example we simulate without any input uncertainty and with unbiased estimator of distance and time
P_c = 10 * norm(MU_d, SIGMA_d) + 5 * norm(MU_t, SIGMA_t)
P_f = 10 * norm(MU_d, SIGMA_d) + 5 * norm(MU_t, SIGMA_t)
C(P_c) = e^(-0.01*P_c)
"""
import numpy as np
import matplotlib.pyplot as plt
import datetime

start = datetime.datetime.now()

MU_d = 5
SIGMA_d = 1

MU_t = 5
SIGMA_t = 1


def one_run():
    K = 10000

    global MU_d
    global MU_t
    global SIGMA_d
    global SIGMA_t

    P_c = 10 * np.random.normal(MU_d, 0, K) + 5 * np.random.normal(MU_t, 0, K)
    P_f = 10 * np.random.normal(MU_d, SIGMA_d, K) + 5 * np.random.normal(MU_t, SIGMA_t, K)

    exp_profit = P_c - P_f
    conv_prob = np.exp(-0.01 * P_c)
    unif = np.random.random_sample(K)

    binom = conv_prob >= unif

    profit = np.ma.masked_equal(binom * exp_profit, 0, copy=True)

    return np.mean(profit)


all_runs = []

for i in range(1000):
    all_runs.append(one_run())


print(all_runs)
print(len(all_runs))
print(max(all_runs))

num_bins = 50

fig, ax = plt.subplots()

n, bins, patches = ax.hist(all_runs, num_bins, normed=0)

ax.set_xlabel('profit')
ax.set_ylabel('count')
ax.set_title('Unbiased Pricing & Exp Demand, Mu_d = %d, S_d = %d, Mu_t = %d, S_t = %d' %(MU_d, SIGMA_d, MU_t, SIGMA_t))

print("ready")

end = datetime.datetime.now()

print(end-start)

plt.show()

