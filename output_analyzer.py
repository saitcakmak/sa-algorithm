import numpy as np


def out_collector(rho, alpha):
    rho = rho + "_"
    alpha = str(alpha)

    mid = "_run_"
    rest = "_iter_1000_eps20-100_x.npy"

    out_list = []
    for i in range(1, 51):
        x_list = np.load("sa_out/offline/"+rho+alpha+mid+str(i)+rest)
        out_list.append(x_list[-1])
    print("mean: ", np.average(out_list), " std: ", np.std(out_list))
    return out_list, np.average(out_list), np.std(out_list)
