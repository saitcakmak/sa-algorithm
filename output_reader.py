"""
read the SA output for the given runs and plot it if needed
"""

import numpy as np
import matplotlib.pyplot as plt

rho_list = ['VaR', 'CVaR']
alpha_list = [0.5, 0.6, 0.7, 0.8, 0.9]

text = "_run_1_iter_1000_eps20-100_x.npy"

for i in range(2):
    plt.figure(i+1)
    rho = rho_list[i]
    for alpha in alpha_list:
        vals = np.load("sa_out/offline/"+rho+'_'+str(alpha)+text)
        plt.plot(range(1, len(vals)+1), vals, label=rho+"$_{"+str(alpha)+"}$")
    plt.title(rho)
    plt.xlabel("Iteration")
    plt.ylabel("Price")
    plt.ylim((15, 35))
    plt.legend(loc=2)
plt.show()
