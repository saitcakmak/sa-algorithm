"""
read the SA output for the given runs and plot it if needed
"""

import numpy as np
import matplotlib.pyplot as plt

rho_list = ['VaR', 'CVaR']
alpha_list = [0.5, 0.6, 0.7, 0.8, 0.9]

text = "_run_1_iter_1000_eps20-100_x.npy"

fig, axes = plt.subplots(2, 1, "all", "all", figsize=(8, 6))
y_min = 17
y_max = 33

for i in range(2):
    rho = rho_list[i]
    for alpha in alpha_list:
        vals = np.load("sa_out/offline/"+rho+'_'+str(alpha)+text)
        axes[i].plot(vals, label=rho+"$_{"+str(alpha)+"}$")
    axes[i].legend()
axes[0].set_ylabel("Price")
axes[1].set_ylabel("Price")
axes[1].set_xlabel("Iteration")
plt.ylim(y_min, y_max)
plt.show()
