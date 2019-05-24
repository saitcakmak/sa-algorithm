import numpy as np
import matplotlib.pyplot as plt
import datetime

start = datetime.datetime.now()


def min_dff(n):
    """
    returns the minimum difference up to n samples
    samples are uniform(0,1)
    """
    samples = np.array([np.random.random(n)])
    diff_matrix = np.absolute(samples - samples.transpose()) + np.diag([1] * n)
    diff_list = [1]
    for i in range(1, n):
        diff_list.append(np.minimum(diff_list[i-1], diff_matrix[i, :i]).min())
    return diff_list


if __name__ == "__main__":
    n = 10000
    rep = 100
    diff = np.zeros(n)
    for i in range(rep):
        diff += min_dff(n)
        time = datetime.datetime.now()
        print((i+1), (time-start))

    diff = diff/10

    plt.plot(diff)
    plt.yscale('log')

    plt.show()

    np.save("output/diff.npy", diff)


