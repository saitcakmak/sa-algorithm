import numpy as np
import matplotlib.pyplot as plt

in_data = np.load("../input_data/input_data_online.npy").item()
in_cust = in_data["cust"]
in_prov = in_data["prov"]

size_list = []
c_mean_list = []
c_std_list = []
c_mle_list = []
p_mean_list = []
p_std_list = []
p_mle_list = []
for i in range(1, 31):
    size = 50 * i
    c_rate = 1 / np.average(in_cust[:size])
    p_rate = 1 / np.average(in_prov[:size])
    c_mle = np.log(80 / c_rate - 1) / 10
    p_mle = - np.log((1 - p_rate / 20) / (1 + p_rate / 20)) / 10
    c_list = np.load("mcmc_out/out_c_online_" + str(i) + ".npy")
    p_list = np.load("mcmc_out/out_p_online_" + str(i) + ".npy")
    # plt.plot(c_list)
    # plt.plot(p_list)
    # plt.show()
    c_mean = np.average(c_list)
    c_std = np.std(c_list)
    p_mean = np.average(p_list)
    p_std = np.std(p_list)
    # print(size, c_mean, c_std, c_mle, p_mean, p_std, p_mle)
    size_list.append(size)
    c_mean_list.append(c_mean)
    c_std_list.append(c_std)
    c_mle_list.append(c_mle)
    p_mean_list.append(p_mean)
    p_std_list.append(p_std)
    p_mle_list.append(p_mle)

x = range(1, 31)
plt.figure(figsize=(8, 6))
plt.errorbar(x, c_mean_list, yerr=c_std_list, capsize=4, label="$\\theta^C_{MCMC}$")
plt.plot(x, c_mle_list, label="$\\theta^C_{MLE}$")
plt.errorbar(x, p_mean_list, yerr=p_std_list, capsize=4, label="$\\theta^P_{MCMC}$")
plt.plot(x, p_mle_list, label="$\\theta^P_{MLE}$")
plt.ylabel("$\\theta$")
plt.xlabel("Day")
plt.legend()
plt.show()
