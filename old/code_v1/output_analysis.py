import datetime

start = datetime.datetime.now()


# def val_check(x):
#     out = []
#     for i in range(m):
#         val, der = mm1(10, x)
#         out.append(val)
#     return np.average(out)

#
# def quad_check(x):
#     return (x - 1) ** 2 + 1
#
#
# inp = np.load("output/quad_combined_quad_small_N_10_runs_1000_output.npy").item()
# keys = list(inp.keys())[3:]
#
#
# out = {}
# for key in keys:
#     out[key] = np.zeros(1000)
#     for i in range(1000):
#         out[key][i] = quad_check(inp[key][i])
# print("time: ", datetime.datetime.now()-start)
#
# np.save("output/quad_values_small_N_10_runs_1000.npy", out)


# tiny = np.load("output/combined_mm1_tiny_fix_N_10_output.npy").item()
# small = np.load("output/combined_mm1_small_fixed_N_20_output.npy").item()
# med = np.load("output/combined_mm1_med_N_100_output.npy").item()
# large = np.load("output/combined_mm1_large_fixed_N_1000_output.npy").item()

# keys = list(tiny.keys())[3:]


# tiny_out = {}
# # tiny_stats = {}
#
# for key in keys:
#     tiny_out[key] = []
#     for i in range(30):
#         last_30 = tiny[key][i][0][-30:]
#         out = np.average(last_30)
#         tiny_out[key].append(val_check(out))
#     print("tiny key: ", key, " time ", datetime.datetime.now()-start)
#     # tiny_out[key] = np.sort(small_out[key], axis=0)
#     # min = tiny_out[key][0]
#     # max = tiny_out[key][-1]
#     # avg = np.average(tiny_out[key])
#     # std = np.std(tiny_out[key])
#     # tiny_stats[key] = (min, max, avg, std)


# small_out = {}
# # small_stats = {}
#
# for key in keys:
#     small_out[key] = []
#     for i in range(30):
#         last_30 = small[key][i][0][-30:]
#         out = np.average(last_30)
#         small_out[key].append(val_check(out))
#     print("small key: ", key, " time ", datetime.datetime.now()-start)
    # small_out[key] = np.sort(small_out[key], axis=0)
    # min = small_out[key][0]
    # max = small_out[key][-1]
    # avg = np.average(small_out[key])
    # std = np.std(small_out[key])
    # small_stats[key] = (min, max, avg, std)


# med_out = {}
# # med_stats = {}
#
# for key in keys:
#     med_out[key] = []
#     for i in range(30):
#         last_30 = med[key][i][0][-30:]
#         out = np.average(last_30)
#         med_out[key].append(val_check(out))
#     print("med key: ", key, " time ", datetime.datetime.now()-start)
    # med_out[key] = np.sort(med_out[key], axis=0)
    # min = med_out[key][0]
    # max = med_out[key][-1]
    # avg = np.average(med_out[key])
    # std = np.std(med_out[key])
    # med_stats[key] = (min, max, avg, std)

# large_out = {}
# # large_stats = {}
#
# for key in keys:
#     large_out[key] = []
#     for i in range(30):
#         last_30 = large[key][i][0][-30:]
#         out = np.average(last_30)
#         large_out[key].append(val_check(out))
#     print("large key: ", key, " time ", datetime.datetime.now()-start)
    # large_out[key] = np.sort(large_out[key], axis=0)
    # min = large_out[key][0]
    # max = large_out[key][-1]
    # avg = np.average(large_out[key])
    # std = np.std(large_out[key])
    # large_stats[key] = (min, max, avg, std)

# print("small")
# for key in keys:
#     print(key, " ", small_stats[key])
#
# print("med")
# for key in keys:
#     print(key, " ", med_stats[key])
#
# print("large")
# for key in keys:
#     print(key, " ", large_stats[key])

#
# print("tiny")
# for key in keys:
#     print(key, " ", tiny_out[key])

# print("small")
# for key in keys:
#     print(key, " ", small_out[key])
#
# print("med")
# for key in keys:
#     print(key, " ", med_out[key])
#
# print("large")
# for key in keys:
#     print(key, " ", large_out[key])


# small_val = {}
# med_val = {}
# large_val = {}
#
# for key in keys:
#     small_val[key] = []
#     med_val[key] = []
#     large_val[key] = []
#     for i in range(3):
#         small_val[key].append(val_check(small_stats[key][i]))
#         med_val[key].append(val_check(med_stats[key][i]))
#         large_val[key].append(val_check(large_stats[key][i]))
#     print("key: ", key, " time: ", datetime.datetime.now()-start)
#
#
# print("small")
# for key in keys:
#     print(key, " ", small_val[key])
#
# print("med")
# for key in keys:
#     print(key, " ", med_val[key])
#
# print("large")
# for key in keys:
#     print(key, " ", large_val[key])
#
