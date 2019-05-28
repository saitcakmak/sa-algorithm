import numpy as np
import naive_estimator
import lr_estimator
import sequential_estimator
import sequential_lr_estimator
import datetime

x = 2


def simple_run(estimator, budget, rep, alpha, rho):
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
            inner_reps = np.zeros((budget, 2))
            for k in range(budget):
                print("budget ", budget, " rep ", i, rho, " count ", k, " time ",
                      datetime.datetime.now()-start)
                theta_0 = np.random.normal(0, 1, budget)
                theta_1 = np.random.normal(0, 1, budget)
                theta = np.transpose([theta_0, theta_1])
                inner_reps[k] = estimator(theta, x, budget, alpha, rho)
            results[i] = np.average(inner_reps, 0)

    elif rho == "CVaR":
        results = np.zeros((rep, 2))
        for i in range(rep):
            print("budget ", budget, " rep ", i, rho, " time ",
                  datetime.datetime.now() - start)
            theta_0 = np.random.normal(0, 1, budget)
            theta_1 = np.random.normal(0, 1, budget)
            theta = np.transpose([theta_0, theta_1])
            results[i] = estimator(theta, x, budget, alpha, rho)
    else:
        return 0
    np.savetxt(rho+"_"+estimator_text+"_budget_"+str(budget)+"_rep_"+str(rep)+"_time_"+str(datetime.datetime.now())+".csv",
               X=results, delimiter=";")

    return results


if __name__ == "__main__":
    estimator = input("choose the estimator (naive, lr, seq, seq_lr): ")
    # budget = int(input("choose budget: "))
    budget = 100
    # rep = int(input("replications: "))
    rep = 50
    # alpha = float(input("alpha: "))
    alpha = 0.8
    rho = input("rho (VaR or CVaR): ")
    simple_run(estimator, budget, rep, alpha, rho)
