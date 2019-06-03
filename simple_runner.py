import numpy as np
import naive_estimator
import lr_estimator
import sequential_estimator
import sequential_lr_estimator
import datetime
from multiprocessing import Pool

x = 2


def simple_run(estimator, budget, rep, alpha, rho, count):
    prob = "simple"
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
        m = int((budget+1) ** (1/3))
        n = m
        k = m
        for i in range(rep):
            inner_reps = np.zeros((k, 2))
            for j in range(k):
                print(rho, str(alpha), estimator_text, "budget ", budget, " rep ", i, rho, " count ", j, " time ",
                      datetime.datetime.now()-start)
                theta_0 = np.random.normal(0, 1, n)
                theta_1 = np.random.normal(0, 1, n)
                theta = np.transpose([theta_0, theta_1])
                inner_reps[j] = estimator(theta, x, m, alpha, rho, prob)
            results[i] = np.average(inner_reps, 0)

    elif rho == "CVaR":
        results = np.zeros((rep, 2))
        m = int(budget ** (1/2))
        n = m
        for i in range(rep):
            print(rho, str(alpha), estimator_text, "budget ", budget, " rep ", i, rho, " time ",
                  datetime.datetime.now() - start)
            theta_0 = np.random.normal(0, 1, n)
            theta_1 = np.random.normal(0, 1, n)
            theta = np.transpose([theta_0, theta_1])
            results[i] = estimator(theta, x, m, alpha, rho, prob)
    else:
        return 0
    np.savetxt("simple_output/"+rho+"_"+str(alpha)+"_"+estimator_text+"_budget_"+str(budget)+"_rep_"
               +str(rep)+"_time_"+str(datetime.datetime.now())+str(count)+".csv",
               X=results, delimiter=";")

    return results


if __name__ == "__main__":
    estimator_list = ['lr']
    rho_list = ['VaR', 'CVaR']
    budget_list = [1000, 10000, 100000, 1000000]
    total_rep = 100
    rep_list = total_rep * np.array([1, 1, 1, 0.2])
    repeater = [1, 1, 1, 5]
    alpha_list = [0.5, 0.8, 0.99]

    count = 0
    arg_list = []

    print(simple_run("seq", 100000, 1, 0.8, "CVaR", count))

    # for est in estimator_list:
    #     for rh in rho_list:
    #         for alp in alpha_list:
    #             for i in range(4):
    #                 for j in range(repeater[i]):
    #                     count += 1
    #                     arg_list.append((est, budget_list[i], int(rep_list[i]), alp, rh, count))

    # for i in range(10):
    #     count += 1
    #     arg_list.append(("seq_lr", 10000000, 3, 0.8, "CVaR", count))
    #
    # for bud in [1000]:
    #     for est in estimator_list:
    #         for alp in alpha_list:
    #             count += 1
    #             arg_list.append((est, bud, 100, alp, "VaR", count))
    #
    # for bud in [1000000]:
    #     for est in estimator_list:
    #         for alp in alpha_list:
    #             count += 1
    #             arg_list.append((est, bud, 33, alp, "VaR", count))
    #             count += 1
    #             arg_list.append((est, bud, 33, alp, "VaR", count))
    #             count += 1
    #             arg_list.append((est, bud, 34, alp, "VaR", count))

    # for bud in [10000, 1000]:
    #     for est in ["seq", "seq_lr"]:
    #         for alp in alpha_list:
    #             count += 1
    #             arg_list.append((est, bud, 100, alp, "CVaR", count))

    # print(arg_list)
    # print(count)
    # pool = Pool(count)
    # pool_results = pool.starmap(simple_run, arg_list)
    # pool.close()
    # pool.join()

    # estimator = input("choose the estimator (naive, lr, seq, seq_lr): ")
    # budget = int(input("choose budget: "))
    # budget = 100
    # rep = int(input("replications: "))
    # rep = 50
    # alpha = float(input("alpha: "))
    # alpha = 0.8
    # rho = input("rho (VaR or CVaR): ")
    # simple_run(estimator, budget, rep, alpha, rho)
