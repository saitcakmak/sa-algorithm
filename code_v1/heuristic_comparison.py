import numpy as np
from code_v1 import sequential_estimator, green_lr_estimator, naive_quad, sequential_lr_estimator
import datetime
from multiprocessing import Pool


def main(n, m, k, rep, post_a=100, post_b=100):
    start = datetime.datetime.now()

    results = dict()
    arg_list = []
    for i in range(rep):
        arg_list.append((n, m, k, post_a, post_b))
    pool = Pool()
    results['lr'] = pool.starmap(green_lr_estimator.main, arg_list)
    results['naive'] = pool.starmap(naive_quad.main, arg_list)
    results['sequential'] = pool.starmap(sequential_estimator.main, arg_list)
    results['sequential_lr'] = pool.starmap(sequential_lr_estimator.main, arg_list)
    pool.close()
    pool.join()

    end = datetime.datetime.now()
    print("n: ", n, " m: ", m, " k: ", k, "rep: ", rep)
    print("naive: ", np.average(results['naive'], 0), " std: ", np.std(results['naive'], 0))
    print("lr: ", np.average(results['lr'], 0), " std: ", np.std(results['lr'], 0))
    print("sequential: ", np.average(results['sequential'], 0), " std: ", np.std(results['sequential'], 0))
    print("sequential_lr: ", np.average(results['sequential_lr'], 0), " std: ", np.std(results['sequential_lr'], 0))
    print(end-start)
    return 0


if __name__ == "__main__":
    main(400, 400, 40, 30)



