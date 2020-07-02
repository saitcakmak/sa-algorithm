import numpy as np
import matplotlib.pyplot as plt
import datetime

start = datetime.datetime.now()
out_dict = np.load("rho_output.npy").item()
spacing = 0.3

# This is the val plot
fig, axes = plt.subplots(3, 3, "all", "all", figsize=(8, 6))

rho_list = ["mle", "mean", "mean_variance"]
alpha_list = [0, 0, 0.1]
print_list = ["MLE", "Expectation", "Mean-Variance"]
x_min = -7.5
x_max = 0

for j in range(3):
    rho = rho_list[j]
    alpha = alpha_list[j]
    val_list = out_dict["val"][rho][alpha]

    axes[0, j].hist(val_list, bins=25, range=(x_min, x_max))
    axes[0, j].set_title(print_list[j])


alpha_list = [0.5, 0.7, 0.9]

rho = "VaR"

for j in range(3):
    alpha = alpha_list[j]
    val_list = out_dict["val"][rho][alpha]

    axes[1, j].hist(val_list, bins=25, range=(x_min, x_max))
    axes[1, j].set_title(rho + "$_{" + str(alpha) + "}$")


rho = "CVaR"

for j in range(3):
    alpha = alpha_list[j]
    val_list = out_dict["val"][rho][alpha]

    axes[2, j].hist(val_list, bins=25, range=(x_min, x_max))
    axes[2, j].set_title(rho + "$_{" + str(alpha) + "}$")


fig.subplots_adjust(hspace=spacing)
axes[0, 0].set_ylabel("Count")
axes[1, 0].set_ylabel("Count")
axes[2, 0].set_ylabel("Count")
axes[2, 0].set_xlabel("$H^c(p)$")
axes[2, 1].set_xlabel("$H^c(p)$")
axes[2, 2].set_xlabel("$H^c(p)$")
plt.show()


# This is the x plot
fig, axes = plt.subplots(3, 3, "all", "all", figsize=(8, 6))
bin_count = 40
x_min = 10
x_max = 50

rho_list = ["mle", "mean", "mean_variance"]
alpha_list = [0, 0, 0.1]
print_list = ["MLE", "Expectation", "Mean-Variance"]

for j in range(3):
    rho = rho_list[j]
    alpha = alpha_list[j]
    x_list = out_dict["x"][rho][alpha]
    x_list = np.minimum(x_list, 50)

    axes[0, j].hist(x_list, bins=bin_count, range=(x_min, x_max))
    axes[0, j].set_title(print_list[j])


alpha_list = [0.5, 0.7, 0.9]

rho = "VaR"

for j in range(3):
    alpha = alpha_list[j]
    x_list = out_dict["x"][rho][alpha]
    x_list = np.minimum(x_list, 50)

    axes[1, j].hist(x_list, bins=bin_count, range=(x_min, x_max))
    axes[1, j].set_title(rho + "$_{" + str(alpha) + "}$")


rho = "CVaR"

for j in range(3):
    alpha = alpha_list[j]
    x_list = out_dict["x"][rho][alpha]
    x_list = np.minimum(x_list, 50)

    axes[2, j].hist(x_list, bins=bin_count, range=(x_min, x_max))
    axes[2, j].set_title(rho + "$_{" + str(alpha) + "}$")


fig.subplots_adjust(hspace=spacing)
axes[0, 0].set_ylabel("Count")
axes[1, 0].set_ylabel("Count")
axes[2, 0].set_ylabel("Count")
axes[2, 0].set_xlabel("Price")
axes[2, 1].set_xlabel("Price")
axes[2, 2].set_xlabel("Price")
plt.show()
