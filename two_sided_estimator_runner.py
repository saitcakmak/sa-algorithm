import numpy as np
import naive_estimator
import lr_estimator
import sequential_estimator
import sequential_lr_estimator
import datetime
from multiprocessing import Pool

x = 10
n = 400
m = 40
alpha = 0.6
rep = 2
t_c_list = np.load("mcmc_out/out_c_try.npy")
t_p_list = np.load("mcmc_out/out_p_try.npy")


def run(estimator, rho, count):
    prob = "two_sided"
    np.random.seed()
    estimator_text = estimator
    if estimator == "naive":
        estimator = naive_estimator.estimator
    elif estimator == "lr":
        estimator = lr_estimator.estimator
    elif estimator == "seq":
        estimator = sequential_estimator.estimator
    elif estimator == "seq_lr":
        estimator = sequential_lr_estimator.estimator
    else:
        return 0

    start = datetime.datetime.now()

    if rho == "VaR":
        results = np.zeros((rep, 2))
        for i in range(rep):
            print(rho, str(alpha), estimator_text, "n: ", n, " rep ", i, rho, " time ",
                  datetime.datetime.now()-start)
            index = np.random.randint(100000, size=n)
            t_c = t_c_list[index]
            t_p = t_p_list[index]
            t_list = np.transpose([t_c, t_p])
            results[i] = estimator(t_list, x, m, alpha, rho, prob)

    elif rho == "CVaR":
        results = np.zeros((rep, 2))
        for i in range(rep):
            print(rho, str(alpha), estimator_text, "n: ", n, " rep ", i, rho, " time ",
                  datetime.datetime.now() - start)
            index = np.random.randint(100000, size=n)
            t_c = t_c_list[index]
            t_p = t_p_list[index]
            t_list = np.transpose([t_c, t_p])
            results[i] = estimator(t_list, x, m, alpha, rho, prob)
    else:
        return 0
    np.savetxt("two_sided_estimators/"+rho+"_"+str(alpha)+"_"+estimator_text+"_rep_"
               +str(rep)+"_time_"+str(datetime.datetime.now())+str(count)+".csv",
               X=results, delimiter=";")

    return results


if __name__ == "__main__":
    estimator_list = ['naive', 'lr', 'seq', 'seq_lr']
    rho_list = ['VaR', 'CVaR']

    count = 0
    arg_list = []

    for est in estimator_list:
        for rh in rho_list:
            count += 1
            arg_list.append((est, rh, count))

    print(arg_list)
    print(count)
    pool = Pool(count)
    pool_results = pool.starmap(run, arg_list)
    pool.close()
    pool.join()

