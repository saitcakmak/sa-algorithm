import numpy as np
import naive_estimator
from old.heuristics import sequential_lr_estimator, lr_estimator, sequential_estimator
import datetime
from multiprocessing import Pool

x = 10
t_c_list = np.load("mcmc_out/out_c_try.npy")
t_p_list = np.load("mcmc_out/out_p_try.npy")
t_limit = datetime.timedelta(minutes=3.3)
seq_0 = np.array([0.12, 0.16, 0.22])


def run(estimator, rho, count, n=400, alpha=0.6, rep=100):
    m = int(n/10)
    prob = "two_sided"
    seq = 0
    est_text = estimator
    np.random.seed()
    if estimator == "naive":
        estimator = naive_estimator.estimator
        seq = 0
    elif estimator == "lr":
        estimator = lr_estimator.estimator
        seq = 0
    elif "seq_lr" in estimator:
        seq_ct = estimator[-1]
        estimator = sequential_lr_estimator.estimator
        if seq_ct == "r":
            seq = seq_0
        elif seq_ct == "2":
            seq = np.array([0.12, 0.16])
        elif seq_ct == "3":
            seq = np.array([0.10])
        elif seq_ct == "4":
            seq = np.array([0.04, 0.06])
        elif seq_ct == "5":
            seq = np.array([0.10, 0.15])
    elif "seq" in estimator:
        seq_ct = estimator[-1]
        estimator = sequential_estimator.estimator
        if seq_ct == "q":
            seq = seq_0
        elif seq_ct == "2":
            seq = np.array([0.12, 0.16])
        elif seq_ct == "3":
            seq = np.array([0.10])
        elif seq_ct == "4":
            seq = np.array([0.04, 0.06])
        elif seq_ct == "5":
            seq = np.array([0.10, 0.15])
    else:
        return 0

    start = datetime.datetime.now()

    if rho == "VaR":
        results = np.zeros((rep, 2))
        for i in range(rep):
            now = datetime.datetime.now()
            print(rho, str(alpha), est_text, "n: ", n, " rep ", i, rho, " time ",
                  now-start)
            index = np.random.randint(100000, size=n)
            t_c = t_c_list[index]
            t_p = t_p_list[index]
            t_list = np.transpose([t_c, t_p])
            results[i] = estimator(t_list, x, m, alpha, rho, prob, seq)
            now_2 = datetime.datetime.now()
            if (now_2 - now) > t_limit:
                print("TIME EXCEEDED: ", rho, str(alpha), estimator_text, "n: ", n)
                return 0

    elif rho == "CVaR":
        results = np.zeros((rep, 2))
        for i in range(rep):
            now = datetime.datetime.now()
            print(rho, str(alpha), est_text, "n: ", n, " rep ", i, rho, " time ",
                  now - start)
            index = np.random.randint(100000, size=n)
            t_c = t_c_list[index]
            t_p = t_p_list[index]
            t_list = np.transpose([t_c, t_p])
            results[i] = estimator(t_list, x, m, alpha, rho, prob, seq)
            now_2 = datetime.datetime.now()
            if (now_2 - now) > t_limit:
                print("TIME EXCEEDED: ", rho, str(alpha), estimator_text, "n: ", n)
                return 0

    else:
        return 0

    return results


if __name__ == "__main__":
    estimator_list = ['naive', 'seq', 'seq2', 'seq3', 'seq4', 'seq5']
    rho_list = ['VaR', 'CVaR']
    n_list = [4000, 10000]
    alp = 0.8

    # est = input("estimator: ")
    # rh = input("rho: ")
    # bud = int(input("n: "))
    ct = 0

    for bud in n_list:
        for est in estimator_list:
            for rh in rho_list:
                ct += 1
                arg_list = []
                parts = 20
                rep2 = int(100/parts)
                for i in range(parts):
                    arg_list.append((est, rh, ct, bud, alp, rep2))

                pool = Pool(parts)
                pool_results = pool.starmap(run, arg_list)
                pool.close()
                pool.join()

                res = np.zeros((100, 2))

                print(pool_results)

                for i in range(parts):
                    res[i * rep2: i * rep2 + rep2] = pool_results[i]

                # print(res)

                estimator_text = est

                np.savetxt("two_sided_estimators/"+rh+"_"+str(alp)+"_"+estimator_text+"_n_"
                           +str(bud)+"_time_"+str(datetime.datetime.now())+str(ct)+".csv",
                           X=res, delimiter=";")
