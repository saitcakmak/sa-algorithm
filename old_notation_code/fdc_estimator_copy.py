import numpy as np
import matplotlib.pyplot as plt
import datetime
from mm1_toy import queue


start = datetime.datetime.now()

mu_lambda = 5
std_lambda = 0.1
n, m = 500, 50
rep = 1000
delta = 0.01
max = 2**32-1
alpha = 0.9
min_ind = int(n * alpha)
MU = 6


def sample_cvar(mu):
    mu_1 = mu - delta
    mu_2 = mu + delta

    H_list_1 = []
    H_list_2 = []

    for i in range(n):
        lam = np.random.randn() * std_lambda + mu_lambda

        list_1 = []
        list_2 = []

        seed = np.random.randint(0, max, m)
        for j in range(m):
            out_1, trash = queue(lam, mu_1, seed[j])
            out_2, trash = queue(lam, mu_2, seed[j])
            list_1.append(out_1)
            list_2.append(out_2)

        H_list_1.append(np.average(list_1))
        H_list_2.append(np.average(list_2))

    sorted_1 = np.sort(H_list_1)
    sorted_2 = np.sort(H_list_2)

    cvar_1 = np.average(sorted_1[min_ind:])
    cvar_2 = np.average(sorted_2[min_ind:])

    return (cvar_2 - cvar_1)/(2 * delta)


est_list = []
for i in range(rep):
    print("rep: ", i, " time_est: ", datetime.datetime.now() - start)
    est_list.append(sample_cvar(MU))
    if i%10 == 0:
        print("der = ", np.average(est_list))
        print("std = ", np.std(est_list))

print("done")

print("der = ", np.average(est_list))
print("std = ", np.std(est_list))


end = datetime.datetime.now()

print(end-start)
