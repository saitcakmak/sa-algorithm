import numpy as np
from problem_sampler import two_sided_sampler
import datetime


out = {"x": {}, "val": {}}

start = datetime.datetime.now()
alpha_list = [0.5, 0.7, 0.9]
theta = [0.1, 0.05]

n = 10000

rho = "CVaR"

out["x"][rho] = {}
out["val"][rho] = {}
for j in range(3):
    alpha = alpha_list[j]
    x_list = []
    val_list = []
    for i in range(1, 51):
        np.random.seed(0)
        file_name = rho + "_" + str(alpha) + "_input_" + str(i) + "_iter_1000_eps20-100_x.npy"
        inner_out = np.load("sa_out/rho/" + file_name)
        x = inner_out[-1]
        vals, ders = two_sided_sampler(theta, x, n)
        x_list.append(x)
        val_list.append(np.average(vals))
        now = datetime.datetime.now()
        print(rho + str(alpha) + " i: ", i, " time: ", now-start)
    out["x"][rho][alpha] = x_list
    out["val"][rho][alpha] = val_list


rho = "VaR"

out["x"][rho] = {}
out["val"][rho] = {}
for j in range(3):
    alpha = alpha_list[j]
    x_list = []
    val_list = []
    for i in range(1, 51):
        np.random.seed(0)
        file_name = rho + "_" + str(alpha) + "_input_" + str(i) + "_iter_1000_eps20-100_x.npy"
        inner_out = np.load("sa_out/rho/" + file_name)
        x = inner_out[-1]
        vals, ders = two_sided_sampler(theta, x, n)
        x_list.append(x)
        val_list.append(np.average(vals))
        now = datetime.datetime.now()
        print(rho + str(alpha) + " i: ", i, " time: ", now-start)
    out["x"][rho][alpha] = x_list
    out["val"][rho][alpha] = val_list


rho = "mean_VaR"

out["x"][rho] = {}
out["val"][rho] = {}
for j in range(3):
    alpha = alpha_list[j]
    x_list = []
    val_list = []
    for i in range(1, 51):
        np.random.seed(0)
        file_name = "m_v" + "_" + str(alpha) + "_input_" + str(i) + "_iter_1000_eps20-100_x.npy"
        inner_out = np.load("sa_out/rho/" + file_name)
        x = inner_out[-1]
        vals, ders = two_sided_sampler(theta, x, n)
        x_list.append(x)
        val_list.append(np.average(vals))
        now = datetime.datetime.now()
        print(rho + str(alpha) + " i: ", i, " time: ", now-start)
    out["x"][rho][alpha] = x_list
    out["val"][rho][alpha] = val_list


rho = "mean_CVaR"

out["x"][rho] = {}
out["val"][rho] = {}
for j in range(3):
    alpha = alpha_list[j]
    x_list = []
    val_list = []
    for i in range(1, 51):
        np.random.seed(0)
        file_name = "m_c" + "_" + str(alpha) + "_input_" + str(i) + "_iter_1000_eps20-100_x.npy"
        inner_out = np.load("sa_out/rho/" + file_name)
        x = inner_out[-1]
        vals, ders = two_sided_sampler(theta, x, n)
        x_list.append(x)
        val_list.append(np.average(vals))
        now = datetime.datetime.now()
        print(rho + str(alpha) + " i: ", i, " time: ", now-start)
    out["x"][rho][alpha] = x_list
    out["val"][rho][alpha] = val_list


rho = "mean"

out["x"][rho] = {}
out["val"][rho] = {}
for j in range(1):
    alpha = 0.0
    x_list = []
    val_list = []
    for i in range(1, 51):
        np.random.seed(0)
        file_name = rho + "_" + str(alpha) + "_input_" + str(i) + "_iter_1000_eps20-100_x.npy"
        inner_out = np.load("sa_out/rho/" + file_name)
        x = inner_out[-1]
        vals, ders = two_sided_sampler(theta, x, n)
        x_list.append(x)
        val_list.append(np.average(vals))
        now = datetime.datetime.now()
        print(rho + str(alpha) + " i: ", i, " time: ", now-start)
    out["x"][rho][alpha] = x_list
    out["val"][rho][alpha] = val_list


rho = "mle"

out["x"][rho] = {}
out["val"][rho] = {}
for j in range(1):
    alpha = 0.0
    x_list = []
    val_list = []
    for i in range(1, 51):
        np.random.seed(0)
        file_name = rho + "_" + str(alpha) + "_input_" + str(i) + "_iter_1000_eps20-100_x.npy"
        inner_out = np.load("sa_out/rho/" + file_name)
        x = inner_out[-1]
        vals, ders = two_sided_sampler(theta, x, n)
        x_list.append(x)
        val_list.append(np.average(vals))
        now = datetime.datetime.now()
        print(rho + str(alpha) + " i: ", i, " time: ", now-start)
    out["x"][rho][alpha] = x_list
    out["val"][rho][alpha] = val_list


rho = "mean_variance"

out["x"][rho] = {}
out["val"][rho] = {}
for j in range(2):
    alpha = [0.1, 1.0][j]
    x_list = []
    val_list = []
    for i in range(1, 51):
        np.random.seed(0)
        file_name = rho + "_" + str(alpha) + "_input_" + str(i) + "_iter_1000_eps20-100_x.npy"
        inner_out = np.load("sa_out/rho/" + file_name)
        x = inner_out[-1]
        vals, ders = two_sided_sampler(theta, x, n)
        x_list.append(x)
        val_list.append(np.average(vals))
        now = datetime.datetime.now()
        print(rho + str(alpha) + " i: ", i, " time: ", now-start)
    out["x"][rho][alpha] = x_list
    out["val"][rho][alpha] = val_list


np.save("rho_output.npy", out)