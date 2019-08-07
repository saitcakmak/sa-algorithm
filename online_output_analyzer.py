import numpy as np


rho_list = ["VaR", "CVaR"]
alpha_list = [0.5, 0.6, 0.7, 0.8, 0.9]

out_list = []
for rho in rho_list:
    for alpha in alpha_list:
        out = np.load("sa_out/online_" + rho + "_" + str(alpha) + "__iter_2000_eps20-100_x.npy")
        inner_out = []
        for i in range(1, 11):
            inner_out.append(out[i * 200])
        print(inner_out)
        # out_list.append(inner_out)
# print(out_list)
