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
        pdf = theta * np.exp(- theta * xi)
        data[theta] = [val, der, xi, pdf]
    return 0


def calculate_lr(theta_list, n):
    # TODO: This could be parallelized
    vals = np.zeros(n)
    ders = np.zeros(n)
    for i in range(n):
        theta = theta_list[i]
        weighted_val = []
        weighted_der = []
        for entry in data.values():
            lr = theta * np.exp(- theta * entry[2]) / entry[3]
            val = entry[0] * lr
            der = entry[1] * lr
            weighted_val = weighted_val + val.tolist()
            weighted_der = weighted_der + der.tolist()
        vals[i] = np.average(weighted_val)
        ders[i] = np.average(weighted_der)
    return vals, ders


def main(n=0, m=0, k=0, post_a=100, post_b=100):
    global data
    data = dict()
    if not n * m:
        n = int(input("n: "))
        m = int(input("m: "))
    x = 1  # float(input("x: "))
    start = datetime.datetime.now()
    theta_list = np.random.gamma(post_a, 1 / post_b, n)
    collect_samples(theta_list, m, x)
    vals, ders = calculate_lr(theta_list, n)
    # print("avg_val: ", np.average(vals), " avg_der: ", np.average(ders))
    ind = np.argsort(vals)
    vals = vals[ind]
    ders = ders[ind]
    # print("VaR0.9_val:", vals[int(n * 0.9)], " VaR0.9_der:", ders[int(n * 0.9)])
    end = datetime.datetime.now()
    # print("time: ", end-start, " budget_used: ", n*m)
    budget = n*m
    lr_budget = n * n * m
    return vals[int(n * 0.9)], ders[int(n * 0.9)], budget, lr_budget


if __name__ == "__main__":
    main(100, 100)
