import numpy as np


dct = np.load("output/quad_values_large_N_1000_runs_1000.npy").item()

key_list = list(dct.keys())

for key in key_list:
    # print(key)
    # if key != 'empirical':
    #     lst = np.sort(dct[key] - dct['empirical'])
    # else:
    lst = np.sort(dct[key])
    mean = np.round(np.average(lst), decimals=4)
    std = np.round(np.std(lst), decimals=4)
    q1 = np.round(lst[99], decimals=4)
    q2 = np.round(lst[249], decimals=4)
    q3 = np.round(lst[499], decimals=4)
    q4 = np.round(lst[749], decimals=4)
    q5 = np.round(lst[999], decimals=4)
    print(str(mean), "\t", str(std), "\t", str(q1), "\t", str(q2), "\t", str(q3), "\t", str(q4), "\t", str(q5))
