import datetime
from multiprocessing import Pool as ThreadPool
import numpy as np


def quad(m, theta, x):
    xi = (-1/theta) * np.log(np.random.random(m))
    c = x - xi
    val = np.square(c)
    der = 2 * c
    return val, der, xi


def collect_samples(theta_list, m, x):
    global data
    for theta in theta_list:
        val, der, xi = quad(m, theta, x)
        data[theta][0] += val.tolist()
        data[theta][1] += der.tolist()
        data[theta][2] += xi.tolist()
        data[theta][3] += (theta * np.exp(- theta * xi)).tolist()
    return 0


def sequential_sampler(theta_list, n, m, k, x):
    # TODO: don't save the samples, save memory by saving summary stats
    vals = np.zeros(n)
    ders = np.zeros(n)
    sigma = np.zeros(n)
    budget_used = 0
    sampler_list = theta_list
    for i in range(k):
        collect_samples(sampler_list, int(m/k), x)
        budget_used += len(sampler_list) * int(m/k)

        for j in range(n):
            if theta_list[j] in sampler_list:
                vals[j] = np.average(data[theta_list[j]][0])
                # ders[j] = np.average(data[theta_list[j]][1])
                sigma[j] = np.std(data[theta_list[j]][0])

        ind = np.argsort(vals)
        vals = vals[ind]
        # ders = ders[ind]
        sigma = sigma[ind]
        theta_list = theta_list[ind]

        quant_val = vals[int(n * 0.9)]
        temp = sigma[int(n * 0.9)] ** 2 / len(data[theta_list[int(n * 0.9)]][0])

        sampler_list = []
        # Welch's t-test without the degrees of freedom
        for j in range(n):
            dev = np.sqrt(temp + sigma[j] ** 2 / len(data[theta_list[j]][0]))
            # dif = abs(quant_val - vals[j])
            # print(dev, dif, temp, len(data[theta_list[j]][0]))
            if abs(quant_val - vals[j]) <= 3 * dev:
                sampler_list.append(theta_list[j])
    return budget_used


def calculate_lr(theta_list, n):
    vals = np.zeros(n)
    ders = np.zeros(n)
    lr_budget_used = 0
    lr_sum = 0
    for i in range(n):
        theta = theta_list[i]
        weighted_val = []
        weighted_der = []
        for entry in data.values():
            lr = theta * np.exp(- theta * np.array(entry[2])) / np.array(entry[3])
            lr_sum += lr
            val = np.array(entry[0]) * lr
            der = np.array(entry[1]) * lr
            weighted_val = weighted_val + val.tolist()
            weighted_der = weighted_der + der.tolist()
        lr_budget_used += len(weighted_val)
        vals[i] = np.average(weighted_val)
        ders[i] = np.average(weighted_der)
    ind = np.argsort(vals)
    vals = vals[ind]
    ders = ders[ind]
    return vals / lr_sum, ders / lr_sum, lr_budget_used


def main(n=0, m=0, k=10, post_a=100, post_b=100):
    global data
    data = dict()
    if not n * m:
        n = int(input("n: "))
        m = int(input("m: "))
        k = int(input("k: "))
    x = 1  # float(input("x: "))
    start = datetime.datetime.now()
    theta_list = np.random.gamma(post_a, 1 / post_b, n)
    for theta in theta_list:
        data[theta] = [[], [], [], []]
    budget_used = sequential_sampler(theta_list, n, m, k, x)
    vals, ders, lr_budget_used = calculate_lr(theta_list, n)
    # print("val: ", np.average(vals), " der: ", np.average(ders))
    # print("VaR0.9_val:", vals[int(n * 0.9)], " VaR0.9_der:", ders[int(n * 0.9)])
    end = datetime.datetime.now()
    # print("time: ", end-start, " budget_used: ", budget_used, " lr_budget_used: ", lr_budget_used)
    budget = budget_used
    lr_budget = lr_budget_used
    return vals[int(n * 0.9)], ders[int(n * 0.9)], budget, lr_budget


if __name__ == "__main__":
    main(100, 100)
