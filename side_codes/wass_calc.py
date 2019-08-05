import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

t_c_list = np.load("mcmc_out/out_c_1.npy")
t_p_list = np.load("mcmc_out/out_p_1.npy")

# std = np.std(t_c_list)

n = 10000
# normal_rv_1 = np.random.normal(0, 1, n)
# normal_rv_2 = np.random.normal(0, 1, n)
#
# print("normal(0,1) for reference: ", wasserstein_distance(normal_rv_1, normal_rv_2))
# print("first and last for c: ", wasserstein_distance(t_c_list[:n], t_c_list[-n:]))
# print("first and last for p: ", wasserstein_distance(t_p_list[:n], t_p_list[-n:]))
# print("last 2 blocks for c: ", wasserstein_distance(t_c_list[-2*n:-n], t_c_list[-n:]))
# print("last 2 blocks for p: ", wasserstein_distance(t_p_list[-2*n:-n], t_p_list[-n:]))

c_list = np.zeros((10, 10))
p_list = np.zeros((10, 10))
for i in range(10):
    for j in range(10):
        c_list[i][j] = wasserstein_distance(t_c_list[i*n:i*n+n], t_c_list[j*n:j*n+n])
        p_list[i][j] = wasserstein_distance(t_p_list[i*n:i*n+n], t_p_list[j*n:j*n+n])

print("c_list:")
print(c_list)

print("p_list")
print(p_list)



# plt.plot(t_c_list)
# plt.plot(t_p_list)
# plt.show()

