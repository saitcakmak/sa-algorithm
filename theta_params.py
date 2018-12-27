import numpy as np

std_dev = np.diag([1.5, 3, 0.5])
corr = [[1, 0.2, -0.05], [0.2, 1, 0.3], [-0.05, 0.3, 1]]

mu_theta = [15, 25, 10]
std_theta = np.matmul(np.matmul(std_dev, corr), std_dev)
