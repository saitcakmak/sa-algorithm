import numpy as np
import matplotlib.pyplot as plt
import datetime
from prod_inv import prod


# a = np.load("output/quad_combined_quad_small_N_10_runs_10_output.npy").item()
# b = np.load("output/quad_combined_quad_small2_N_10_runs_90_output.npy").item()
# c = np.load("output/quad_combined_quad_small3_N_10_runs_100_output.npy").item()
# d = np.load("output/quad_combined_quad_small4_N_10_runs_100_output.npy").item()
#
# e = {}
#
# for key in a.keys():
#     e[key] = a[key] + b[key] + c[key] + d[key]
#
# np.save("output/quad_combined_quad_small_N_10_runs_300_output.npy", e)

# def quad(m, theta, x):
#     xi = (-1/theta) * np.log(np.random.random(m))
#     c = x - xi
#     val = np.inner(c, c) / m
#     der = 2 * np.average(c)
#     return val, der
#
#
# def quad_vectorized(n, m, theta, x):
#     xi = np.multiply((-1/theta), np.log(np.random.random((m, n))))
#     c = x - xi
#     val = np.linalg.norm(c, axis=0) ** 2 / m
#     der = 2 * np.average(c, axis=0)
#     return val, der
#
#
# def collect_inner_samples(m, theta, x):
#     m = int(m)
#     return quad(m, theta, x)
#
#
# def collect_samples(n, m, x):
#     sample_list = np.zeros(n)
#     derivative_list = np.zeros(n)
#     theta = np.random.gamma(post_a, 1 / post_b, n)
#     for i in range(n):
#         val, der = collect_inner_samples(m, theta[i], x)
#         sample_list[i] = val
#         derivative_list[i] = der
#     return np.array(sample_list), np.array(derivative_list)
#
#
# def collect_samples_v2(n, m, x):
#     theta = np.random.gamma(post_a, 1/post_b, n)
#     val, der = quad_vectorized(n, m, theta, x)
#     return val, der
#
#
# post_a = 1000
# post_b = 1000
#
#
# start = datetime.datetime.now()
# for i in range(1):
#     for n in range(100, 10001, 10):
#         collect_samples(n, n/5, 1)
# end = datetime.datetime.now()
# print("v1: ", (end - start))
#
# start = datetime.datetime.now()
# for i in range(1):
#     for n in range(100, 10001, 10):
#         collect_samples(n, n / 5, 1)
# end = datetime.datetime.now()
# print("v2: ", (end - start))




# x = np.arange(-5, 15, 0.1)
# theta = 1
# res = []
# for i in x:
#     # np.random.seed()
#     # out_l = []
#     print("i: ", i)
#     # for j in range(4000):
#     #     out, der = prod(theta, i)
#     #     out_l.append(out)
#     # res.append(np.average(out_l))
#     res.append(quad(400000, 2, i)[0])
#
# plt.plot(x, res)
# plt.show()
