import datetime
import numpy as np


def quad(m, theta, x):
    xi = (-1/theta) * np.log(np.random.random(m))
    c = x - xi
    val = np.inner(c, c) / m
    der = 2 * np.average(c)
    return val, der


def collect_samples(n, m, x, post_a, post_b):
    sample_list = np.zeros(n)
    derivative_list = np.zeros(n)
    theta = np.random.gamma(post_a, 1 / post_b, n)
    for i in range(n):
        val, der = quad(m, theta[i], x)
        sample_list[i] = val
        derivative_list[i] = der
    return np.array(sample_list), np.array(derivative_list)


def main(n=0, m=0, k=0, post_a=100, post_b=100):
    if not n * m:
        n = int(input("n: "))
        m = int(input("m: "))
    x = 1  # float(input("x: "))
    start = datetime.datetime.now()
    vals, ders = collect_samples(n, m, x, post_a, post_b)
    # print("avg_val: ", np.average(vals), " avg_der: ", np.average(ders))
    ind = np.argsort(vals)
    vals = vals[ind]
    ders = ders[ind]
    # print("VaR0.9_val:", vals[int(n * 0.9)], " VaR0.9_der:", ders[int(n * 0.9)])
    end = datetime.datetime.now()
    # print("time: ", end - start, " budget_used: ", n * m)
    budget = n*m
    lr_budget = 0
    return vals[int(n * 0.9)], ders[int(n * 0.9)], budget, lr_budget


if __name__ == "__main__":
    main(100, 100)