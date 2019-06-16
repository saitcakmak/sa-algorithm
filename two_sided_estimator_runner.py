import numpy as np
import naive_estimator
import lr_estimator
import sequential_estimator
import sequential_lr_estimator
import datetime
from multiprocessing import Pool

x = 10
t_c_list = np.load("mcmc_out/out_c_try.npy")
t_p_list = np.load("mcmc_out/out_p_try.npy")
t_limit = datetime.timedelta(minutes=3.3)


def run(estimator, rho, count, n=400, alpha=0.6, rep=100):
    m = int(n/10)
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
            now = datetime.datetime.now()
            print(rho, str(alpha), estimator_text, "n: ", n, " rep ", i, rho, " time ",
                  now-start)
            index = np.random.randint(100000, size=n)
            t_c = t_c_list[index]
            t_p = t_p_list[index]
            t_list = np.transpose([t_c, t_p])
            results[i] = estimator(t_list, x, m, alpha, rho, prob)
            now_2 = datetime.datetime.now()
            if (now_2 - now) > t_limit:
                print("TIME EXCEEDED: ", rho, str(alpha), estimator_text, "n: ", n)
                return 0

    elif rho == "CVaR":
        results = np.zeros((rep, 2))
        for i in range(rep):
            now = datetime.datetime.now()
            print(rho, str(alpha), estimator_text, "n: ", n, " rep ", i, rho, " time ",
                  now - start)
            index = np.random.randint(100000, size=n)
            t_c = t_c_list[index]
            t_p = t_p_list[index]
            t_list = np.transpose([t_c, t_p])
            results[i] = estimator(t_list, x, m, alpha, rho, prob)
            now_2 = datetime.datetime.now()
            if (now_2 - now) > t_limit:
                print("TIME EXCEEDED: ", rho, str(alpha), estimator_text, "n: ", n)
                return 0

    else:
        return 0
    np.savetxt("two_sided_estimators/"+rho+"_"+str(alpha)+"_"+estimator_text+"_n_"
               +str(n)+"_time_"+str(datetime.datetime.now())+str(count)+".csv",
               X=results, delimiter=";")

    return results


if __name__ == "__main__":
    # estimator_list_1 = ['naive', 'lr', 'seq', 'seq_lr']
    # estimator_list_2 = ['naive', 'seq']
    # rho_list = ['VaR', 'CVaR']
    # n_list = [100, 400, 1000, 4000, 10000, 40000, 100000]
    alpha = 0.7

    estimator = input("estimator: ")
    rho = int("rho: ")
    n = int(input("n: "))
    count = 0
    run(estimator, rho, count, n, alpha, 100)
    #
    # count = 0
    # arg_list = []
    #
    # for n in n_list:
    #     if n < 10000:
    #         for est in estimator_list_1:
    #             for rh in rho_list:
    #                 count += 1
    #                 arg_list.append((est, rh, count, n, alpha, 100))
    #     else:
    #         for est in estimator_list_2:
    #             for rh in rho_list:
    #                 count += 1
    #                 arg_list.append((est, rh, count, n, alpha, 100))
    #
    # print(arg_list)
    # print(count)
    # pool = Pool(count)
    # pool_results = pool.starmap(run, arg_list)
    # pool.close()
    # pool.join()

