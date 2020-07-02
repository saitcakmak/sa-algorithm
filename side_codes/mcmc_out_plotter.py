import numpy as np
import matplotlib.pyplot as plt


string_list = ["1"]
for string in string_list:
    plt.figure(string)
    data = np.load("../mcmc_out/out_p_" + string + ".npy")
    plt.plot(data, label=string, color='red')


plt.legend()
plt.show()
