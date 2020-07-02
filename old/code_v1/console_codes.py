# count = 1000000
# res = np.zeros(count)
# for i in range(count):
#     x = np.random.randint(0, 20, 100)
#     h = np.zeros(20)
#     for j in range(20):
#         h[j] = np.sum(x == j)
#     res[i] = np.sum(h >= 5)
#
# print(np.average(res))


# a1 = np.load("output/quad_combined_quad_s1_N_10_runs_100_output.npy").item()
# a2 = np.load("output/quad_combined_quad_s2_N_10_runs_100_output.npy").item()
# a3 = np.load("output/quad_combined_quad_s3_N_10_runs_100_output.npy").item()
# a4 = np.load("output/quad_combined_quad_s4_N_10_runs_100_output.npy").item()
# a5 = np.load("output/quad_combined_quad_s5_N_10_runs_100_output.npy").item()
# a6 = np.load("output/quad_combined_quad_s6_N_10_runs_100_output.npy").item()
# a7 = np.load("output/quad_combined_quad_s7_N_10_runs_100_output.npy").item()
# a8 = np.load("output/quad_combined_quad_small_N_10_runs_300_output.npy").item()
# a9 = np.load("output/quad_combined_quad_ss9_N_20_runs_63_output.npy").item()
# a10 = np.load("output/quad_combined_quad_ss10_N_20_runs_63_output.npy").item()
# a11 = np.load("output/quad_combined_quad_ss11_N_20_runs_63_output.npy").item()
# a12 = np.load("output/quad_combined_quad_ss12_N_20_runs_63_output.npy").item()
# a13 = np.load("output/quad_combined_quad_ss13_N_20_runs_63_output.npy").item()
# a14 = np.load("output/quad_combined_quad_ss14_N_20_runs_63_output.npy").item()
# a15 = np.load("output/quad_combined_quad_ss15_N_20_runs_63_output.npy").item()
# a16 = np.load("output/quad_combined_quad_ss16_N_20_runs_63_output.npy").item()
# a17 = np.load("output/quad_combined_quad_mm17_N_100_runs_50_output.npy").item()
# a18 = np.load("output/quad_combined_quad_mm18_N_100_runs_50_output.npy").item()
# a19 = np.load("output/quad_combined_quad_mm19_N_100_runs_50_output.npy").item()
# a20 = np.load("output/quad_combined_quad_mm20_N_100_runs_50_output.npy").item()

# e = {}
#
# for key in a1.keys():
#     e[key] = a1[key] + a2[key] + a3[key] + a4[key] + a5[key] + a6[key] + a7[key] + a8[key]  # + a9[key] + a10[key]
#     # e[key] = e[key] + a11[key] + a12[key] + a13[key] + a14[key] + a15[key] + a16[key]  # + a17[key] + a18[key] + a19[key] + a20[key]
#
# np.save("output/quad_combined_quad_small_N_10_runs_1000_output.npy", e)

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
