import numpy as np
import matplotlib.pyplot as plt

rho_list = ['VaR', 'CVaR']
alpha_list = [0.5, 0.6, 0.7, 0.8, 0.9]

text = "__iter_1000_eps20-100_x.npy"

for i in range(2):
    plt.figure(i+1)
    rho = rho_list[i]
    for alpha in alpha_list:
        # vals = np.load("sa_out/online_"+rho+'_'+str(alpha)+text)
        vals = np.load("sa_out/"+rho+'_'+str(alpha)+text)
        val = vals[-1]
        std = np.std(vals[-50:])
        plt.plot(range(1, len(vals)+1), vals, label=rho+"$_{"+str(alpha)+"}$")
    plt.title(rho)
    plt.xlabel("Iteration")
    plt.ylabel("Price")
    plt.ylim((10, 25))
    plt.legend()
    plt.show()
