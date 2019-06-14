import numpy as np
import matplotlib.pyplot as plt
import datetime
from problem_sampler import two_sided_sampler as sampler
from multiprocessing import Pool


def plotter(theta, begin=10, end=30, step=0.1, m=400, count=0):
    start = datetime.datetime.now()
    x_vals = np.arange(begin, end, step)
    true_out = []
    for x in x_vals:
        true_out.append(np.average(sampler(theta, x, m)[0]))
        end = datetime.datetime.now()
        print("time: ", end - start, " x: ", x)
    end = datetime.datetime.now()
    print("time: ", end-start, " count: ", count)
    return x_vals, true_out

