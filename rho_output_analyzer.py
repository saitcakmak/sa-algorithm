import numpy as np
import matplotlib.pyplot as plt
import datetime

start = datetime.datetime.now()
out_dict = np.load("rho_output.npy").item()

# This is the val plot
fig, axes = plt.subplots(5, 3, "all", "all", figsize=(8, 6))

rho_list = ["mle", "mean", "mean_variance"]
alpha_list = [0, 0, 0.1]
print_list = ["MLE", "Expectation", "Mean-Variance"]

for j in range(3):
    rho = rho_list[j]
    alpha = alpha_list[j]
    x_list = out_dict["x"][rho][alpha]
    val_list = out_dict["val"][rho][alpha]

    axes[0, j].hist(val_list, bins=25, range=(-7.5, -1.5))
    axes[0, j].set_title(print_list[j])


alpha_list = [0.5, 0.7, 0.9]

rho = "VaR"

for j in range(3):
    alpha = alpha_list[j]
    x_list = out_dict["x"][rho][alpha]
    val_list = out_dict["val"][rho][alpha]

    axes[1, j].hist(val_list, bins=25, range=(-7.5, -1.5))
    axes[1, j].set_title(rho + "$_{" + str(alpha) + "}$")


rho = "CVaR"

for j in range(3):
    alpha = alpha_list[j]
    x_list = out_dict["x"][rho][alpha]
    val_list = out_dict["val"][rho][alpha]

    axes[2, j].hist(val_list, bins=25, range=(-7.5, -1.5))
    axes[2, j].set_title(rho + "$_{" + str(alpha) + "}$")


rho = "mean_VaR"
print_rho = "Mean-VaR"

for j in range(3):
    alpha = alpha_list[j]
    x_list = out_dict["x"][rho][alpha]
    val_list = out_dict["val"][rho][alpha]

    axes[3, j].hist(val_list, bins=25, range=(-7.5, -1.5))
    axes[3, j].set_title(print_rho + "$_{" + str(alpha) + "}$")

rho = "mean_CVaR"
print_rho = "Mean-CVaR"

for j in range(3):
    alpha = alpha_list[j]
    x_list = out_dict["x"][rho][alpha]
    val_list = out_dict["val"][rho][alpha]

    axes[4, j].hist(val_list, bins=25, range=(-7.5, -1.5))
    axes[4, j].set_title(print_rho + "$_{" + str(alpha) + "}$")

fig.subplots_adjust(hspace=0.6)
plt.show()


# This is the x plot
fig, axes = plt.subplots(5, 3, "all", "all", figsize=(8, 6))
bin_count = 40

rho_list = ["mle", "mean", "mean_variance"]
alpha_list = [0, 0, 0.1]
print_list = ["MLE", "Expectation", "Mean-Variance"]

for j in range(3):
    rho = rho_list[j]
    alpha = alpha_list[j]
    x_list = out_dict["x"][rho][alpha]
    val_list = out_dict["val"][rho][alpha]

    axes[0, j].hist(x_list, bins=bin_count, range=(10, 50))
    axes[0, j].set_title(print_list[j])


alpha_list = [0.5, 0.7, 0.9]

rho = "VaR"

for j in range(3):
    alpha = alpha_list[j]
    x_list = out_dict["x"][rho][alpha]
    val_list = out_dict["val"][rho][alpha]

    axes[1, j].hist(x_list, bins=bin_count, range=(10, 50))
    axes[1, j].set_title(rho + "$_{" + str(alpha) + "}$")


rho = "CVaR"

for j in range(3):
    alpha = alpha_list[j]
    x_list = out_dict["x"][rho][alpha]
    val_list = out_dict["val"][rho][alpha]

    axes[2, j].hist(x_list, bins=bin_count, range=(10, 50))
    axes[2, j].set_title(rho + "$_{" + str(alpha) + "}$")


rho = "mean_VaR"
print_rho = "Mean-VaR"

for j in range(3):
    alpha = alpha_list[j]
    x_list = out_dict["x"][rho][alpha]
    val_list = out_dict["val"][rho][alpha]

    axes[3, j].hist(x_list, bins=bin_count, range=(10, 50))
    axes[3, j].set_title(print_rho + "$_{" + str(alpha) + "}$")

rho = "mean_CVaR"
print_rho = "Mean-CVaR"

for j in range(3):
    alpha = alpha_list[j]
    x_list = out_dict["x"][rho][alpha]
    val_list = out_dict["val"][rho][alpha]

    axes[4, j].hist(x_list, bins=bin_count, range=(10, 50))
    axes[4, j].set_title(print_rho + "$_{" + str(alpha) + "}$")

fig.subplots_adjust(hspace=0.6)
plt.show()