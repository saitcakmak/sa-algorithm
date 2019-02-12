import numpy as np
import matplotlib.pyplot as plt
import datetime


def compare(n, m, x):
    t_1 = np.random.normal(0, 1, n)
    t_2 = np.random.normal(0, 1, n)
    xi = np.random.normal(0, 10/m, n)
    H_t = np.array((x * t_1 + t_2, t_1, t_2))
    H_hat_t = np.array((x * t_1 + t_2 + (x + t_2) * xi, t_1, t_2))
    H_t = H_t[:, H_t[0, :].argsort()]
    H_hat_t = H_hat_t[:, H_hat_t[0, :].argsort()]
    return H_t[:, int(0.9 * n)], H_hat_t[:, int(0.9 * n)]


if __name__ == "__main__":
    # np.random.seed(0)
    start = datetime.datetime.now()
    n = 10000
    rep = 1000
    pow = 0.5
    res_true = np.zeros(rep)
    res_hat = np.zeros(rep)
    for i in range(rep):
        out = compare(n, int(n ** pow), 2)
        res_true[i] = out[0][1]
        res_hat[i] = out[1][1]
    # plt.figure(1)
    plt.hist(res_true, 50, (-2, 4), density=True, cumulative=True, histtype='step', label='true', color='red')
    # plt.figure(2)
    plt.hist(res_hat, 50, (-2, 4), density=True, cumulative=True, histtype='step', label='estimate', color='blue')
    plt.legend(loc=4)
    title = "CDF of $\\theta_1$ with n = " + str(n) + " m = $n^{" + str(pow) + "}$ using " + str(rep) + " replications"
    plt.title(title)
    end = datetime.datetime.now()
    print((end-start))
    plt.show()
