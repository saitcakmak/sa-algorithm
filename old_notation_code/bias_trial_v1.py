import numpy as np
import matplotlib.pyplot as plt
import datetime
from mm1_toy import queue


start = datetime.datetime.now()

mu_lambda = 5
std_lambda = 0.1
n, m = 10000, 50
max = 2**32-1
alpha = 0.9
MU = 6


def collect_samples():
    sample_list = []
    derivative_list =[]

    for i in range(n):
        print(i)
        lam = np.random.randn() * std_lambda + mu_lambda
        inner_list = []
        inner_derivative_list = []

        seed = np.random.randint(0, max, m)
        for j in range(m):
            out, der = queue(lam, MU, seed[j])
            inner_list.append(out)
            inner_derivative_list.append(der)

        sample_list.append(inner_list)
        derivative_list.append(inner_derivative_list)

    return sample_list, derivative_list


def calc_der(sample_list, derivative_list, n_k, m_k):
    mean_list = []
    mean_der_list = []

    for i in range(n_k):
        mean_list.append(np.average(sample_list[i][0:m_k]))
        mean_der_list.append(np.average(derivative_list[i][0:m_k]))
    var_alpha = np.sort(mean_list)[int(n_k * alpha)]

    cvar_der_list = []
    for i in range(n_k):
        if mean_list[i] >= var_alpha:
            cvar_der_list.append(mean_der_list[i])

    return np.average(cvar_der_list)


s_list, d_list = collect_samples()

print("samples ready")

for i in range(100, n+1, 100):
    for j in range(10, m+1, 10):
        print("der M=", i, "m=", j, calc_der(s_list, d_list, i, j))


end = datetime.datetime.now()

print(end-start)

plt.show()