"""
estimate the true value and derivative w.r.t. theta
"""
import numpy as np
import datetime
from multiprocessing import Pool as ThreadPool
from mm1_toy import queue_with_theta_der

start = datetime.datetime.now()

prob = queue_with_theta_der
prob_str = "queue"
count = 1


def collect_inner_samples(m, theta, x):
    global prob, count
    print("count: ", count, " time: ", (datetime.datetime.now() - start))
    count += 1
    inner_list = []
    inner_derivative_list = []
    for j in range(m):
        val, der = prob(theta, x)
        inner_list.append(val)
        inner_derivative_list.append(der)
    return np.average(inner_list), np.average(inner_derivative_list, 0)


def collect_samples(n, m, theta, x):
    sample_list = []
    derivative_list = []
    arg_list = []

    for i in range(n):
        arg_list.append((m, theta, x))
    pool = ThreadPool()
    results = pool.starmap(collect_inner_samples, arg_list)
    pool.close()
    pool.join()
    for res in results:
        sample_list.append(res[0])
        derivative_list.append(res[1])
    return np.average(sample_list), np.average(derivative_list)


if __name__ == "__main__":
    val, der = collect_samples(10000, 1000, 10, 6)
    output = {"val": val, "der": der}
    np.save("10000000_runs_10_6_val_der.npy", output)
    print(output)
