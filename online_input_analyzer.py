import numpy as np

in_data = np.load("input_data/input_data_online.npy").item()
in_cust = in_data["cust"]
in_prov = in_data["prov"]
for i in range(10):
    size = 10 * 2 ** i
    c_rate = 1 / np.average(in_cust[:size])
    p_rate = 1 / np.average(in_prov[:size])
    c_mle = np.log(80 / c_rate - 1) / 10
    p_mle = - np.log((1 - p_rate / 20) / (1 + p_rate / 20)) / 10
    c_list = np.load("mcmc_out/out_c_online_" + str(i) + ".npy")
    p_list = np.load("mcmc_out/out_p_online_" + str(i) + ".npy")
    c_mean = np.average(c_list)
    c_std = np.std(c_list)
    p_mean = np.average(p_list)
    p_std = np.std(p_list)
    print(size, c_mean, c_std, c_mle, p_mean, p_std, p_mle)
