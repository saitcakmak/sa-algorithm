import numpy as np

std_dev = np.diag([3, 0.5, 1.5])
corr = [[1, 0.3, 0.2], [0.3, 1, -0.05], [0.2, -0.05, 1]]

mu_theta = [25, 10, 15]
std_theta = np.matmul(np.matmul(std_dev, corr), std_dev)
