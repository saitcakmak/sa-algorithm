import datetime
from multiprocessing import Pool as ThreadPool
import numpy as np


def quad(m, theta, x):
    xi = (-1/theta) * np.log(np.random.random(m))
    c = x - xi
    val = np.square(c)
    der = 2 * c
    return val, der


def collect_samples(theta_list, m, x):
    global data
    for theta in theta_list:
        val, der = quad(m, theta, x)
        data[theta][0] += val.tolist()
        data[theta][1] += der.tolist()
    return 0


def sequential_sampler(theta_list, n, m, p, x):
    vals = np.zeros(n)
    ders = np.zeros(n)
    sigma = np.zeros(n)
    budget_used = 0
    sampler_list = theta_list
    for i in range(p):
        collect_samples(sampler_list, int(m/p), x)
        budget_used += len(sampler_list) * int(m/p)

        for j in range(n):
            if theta_list[j] in sampler_list:
                vals[j] = np.average(data[theta_list[j]][0])
                ders[j] = np.average(data[theta_list[j]][1])
                sigma[j] = np.std(data[theta_list[j]][0])

        ind = np.argsort(vals)
        vals = vals[ind]
        ders = ders[ind]
        sigma = sigma[ind]
        theta_list = theta_list[ind]

        quant_val = vals[int(n * 0.9)]
        temp = sigma[int(n * 0.9)] ** 2 / len(data[theta_list[int(n * 0.9)]][0])

        sampler_list = []
        # Welch's t-test without the degrees of freedom
        for j in range(n):
            dev = np.sqrt(temp + sigma[j] ** 2 / len(data[theta_list[j]][0]))
            dif = abs(quant_val - vals[j])
            # print(dev, dif, temp, len(data[theta_list[j]][0]))
            if abs(quant_val - vals[j]) <= 3 * dev:
                sampler_list.append(theta_list[j])

    return vals, ders, budget_used


def main(n=0, m=0, p=10, post_a=100, post_b=100):
    global data
    data = dict()
    if not n * m:
        n = int(input("n: "))
        m = int(input("m: "))
        p = int(input("p: "))
    x = 1  # float(input("x: "))
    start = datetime.datetime.now()
    theta_list = np.random.gamma(post_a, 1 / post_b, n)
    for theta in theta_list:
        data[theta] = [[], []]
    vals, ders, budget_used = sequential_sampler(theta_list, n, m, p, x)
    # print("val: ", np.average(vals), " der: ", np.average(ders))
    # print("VaR0.9_val:", vals[int(n * 0.9)], " VaR0.9_der:", ders[int(n * 0.9)])
    end = datetime.datetime.now()
    # print("time: ", end-start, " budget_used: ", budget_used)
    budget = budget_used
    lr_budget = 0
    return vals[int(n * 0.9)], ders[int(n * 0.9)], budget, lr_budget


if __name__ == "__main__":
    main(100, 100)
