import numpy as np
import matplotlib.pyplot as plt


rho_list = ["VaR", "CVaR"]
alpha_list = [0.5, 0.6, 0.7, 0.8, 0.9]

# out_list = []
# for rho in rho_list:
#     for alpha in alpha_list:
#         out = np.load("sa_out/online_" + rho + "_" + str(alpha) + "__iter_3000_eps20-100_x.npy")
#         inner_out = []
#         for i in range(1, 31):
#             inner_out.append(out[i * 100])
#         print(inner_out)
#         # out_list.append(inner_out)
# # print(out_list)

fig, axes = plt.subplots(2, 1, "all", "all", figsize=(8, 6))
y_min = 17
y_max = 27

rho = "VaR"
for alpha in alpha_list:
    out = np.load("sa_out/online_" + rho + "_" + str(alpha) + "__iter_3000_eps20-100_x.npy")
    axes[0].plot(out, label="VaR$_{" + str(alpha) + "}$")

axes[0].legend()

rho = "CVaR"
for alpha in alpha_list:
    out = np.load("sa_out/online_" + rho + "_" + str(alpha) + "__iter_3000_eps20-100_x.npy")
    axes[1].plot(out, label="CVaR$_{" + str(alpha) + "}$")

axes[1].legend()
axes[0].set_ylabel("Price")
axes[1].set_ylabel("Price")
axes[1].set_xlabel("Iteration")
plt.ylim(y_min, y_max)
plt.show()
