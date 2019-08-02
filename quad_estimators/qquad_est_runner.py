import numpy as np
import naive_estimator
from heuristics import lr_estimator, sequential_estimator, sequential_lr_estimator
import datetime
from multiprocessing import Pool


x = 1
t_limit = datetime.timedelta(minutes=1)


def run(estimator, rho, count, n=400, alpha=0.6, rep=100):
    m = n
    prob = "quad"
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
            t_list = np.random.gamma(10, 1/10, n)
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
            t_list = np.random.gamma(10, 1 / 10, n)
            results[i] = estimator(t_list, x, m, alpha, rho, prob)
            now_2 = datetime.datetime.now()
            if (now_2 - now) > t_limit:
                print("TIME EXCEEDED: ", rho, str(alpha), estimator_text, "n: ", n)
                return 0

    else:
        return 0
    # np.savetxt("quad_estimators/"+rho+"_"+str(alpha)+"_"+estimator_text+"_n_"
    #            +str(n)+"_time_"+str(datetime.datetime.now())+str(count)+".csv",
    #            X=results, delimiter=";")

    return results


if __name__ == "__main__":
    # estimator_list = ["seq"]
    # rho_list = ['CVaR']
    # budget_list = [100, 400, 1000, 4000]
    # total_rep = 100
    # rep_list = total_rep * np.array([1, 1, 1, 1])
    # repeater = [1, 1, 1, 1]
    # # alpha_list = [0.5, 0.8, 0.99]
    # alpha_list = [0.9]
    #
    # count = 0
    # arg_list = []
    #
    # for est in estimator_list:
    #     for rh in rho_list:
    #         for alp in alpha_list:
    #             for i in range(4):
    #                 for j in range(repeater[i]):
    #                     count += 1
    #                     # arg_list.append((est, rh, count, budget_list[i], alp, int(rep_list[i])))
    #                     run(est, rh, count, budget_list[i], alp, int(rep_list[i]))

    # print(arg_list)
    # print(count)
    # pool = Pool()
    # pool_results = pool.starmap(run, arg_list)
    # pool.close()
    # pool.join()
    #
    # run("seq_lr", "CVaR", 1, 100, 0.5, 1)

    alpha = 0.9

    estimator = "seq"
    rho = "CVaR"
    n = 10000
    count = 0

    arg_list = []

    for i in range(20):
        arg_list.append((estimator, rho, count, n, alpha, 5))

    pool = Pool()
    pool_results = pool.starmap(run, arg_list)
    pool.close()
    pool.join()

    results = np.zeros((100, 2))

    print(pool_results)

    for i in range(20):
        results[i * 5: i * 5 + 5] = pool_results[i]

    print(results)

    estimator_text = estimator+"5"

    np.savetxt("quad_estimators/"+rho+"_"+str(alpha)+"_"+estimator_text+"_n_"
               +str(n)+"_time_"+str(datetime.datetime.now())+str(count)+".csv",
               X=results, delimiter=";")

