import numpy as np
import sa_runner
from multiprocessing import Pool


run_count = 50
alp = float(input("alpha: "))
rh = input("rho: ")

arg_list = []
for i in range(1, run_count+1):
    text = "input_" + str(i)
    arg_list.append((alp, rh, text, str(i)))


def dummy(alp, rh, text):
    np.random.seed()
    return np.random.rand()


pool = Pool(run_count)
pool_results = pool.starmap(sa_runner.main, arg_list)
# pool_results = pool.starmap(dummy, arg_list)
pool.close()
pool.join()
# print(pool_results)
