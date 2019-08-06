import numpy as np
import datetime
from multiprocessing import Pool


K_c = 40
K_p = 20
price = 10
theta_c = 0.1
theta_p = 0.05

lam_c = K_c * 2 * np.exp(- theta_c * price) / (1 + np.exp(- theta_c * price))
lam_p = K_p * (1 - np.exp(- theta_p * price)) / (1 + np.exp(- theta_p * price))

lb = 0.01
ub = 0.5


def lr_c(candidate, theta, size):
    ratio_list = []
    for entry in samples_c[:size]:
        lam_cand = K_c * 2 * np.exp(- candidate * price) / (1 + np.exp(- candidate * price))
        lam_curr = K_c * 2 * np.exp(- theta * price) / (1 + np.exp(- theta * price))
        p_cand = lam_cand * np.exp(- lam_cand * entry)  # pdf
        p_curr = lam_curr * np.exp(- lam_curr * entry)  # pdf
        ratio_list.append(p_cand/p_curr)
    prob = np.prod(ratio_list)
    return np.nan_to_num(prob)


def lr_p(candidate, theta, size):
    ratio_list = []
    for entry in samples_p[:size]:
        lam_cand = K_p * (1 - np.exp(- candidate * price)) / (1 + np.exp(- candidate * price))
        lam_curr = K_p * (1 - np.exp(- theta * price)) / (1 + np.exp(- theta * price))
        p_cand = lam_cand * np.exp(- lam_cand * entry)  # pdf
        p_curr = lam_curr * np.exp(- lam_curr * entry)  # pdf
        ratio_list.append(p_cand/p_curr)
    prob = np.prod(ratio_list)
    return np.nan_to_num(prob)


def theta_next_c(theta, candidate_std, size):
    """
    return the next theta according to the accept/reject decision
    """
    candidate = np.random.normal(theta, candidate_std)
    accept_prob = lr_c(candidate, theta, size)
    uniform = np.random.random_sample()
    if uniform < accept_prob and ub > candidate > lb:
        return candidate
    else:
        return theta


def theta_next_p(theta, candidate_std, size):
    """
    return the next theta according to the accept/reject decision
    """
    candidate = np.random.normal(theta, candidate_std)
    accept_prob = lr_p(candidate, theta, size)
    uniform = np.random.random_sample()
    if uniform < accept_prob and ub > candidate > lb:
        return candidate
    else:
        return theta


def mcmc_c(run_length, theta, string, candidate_std, size):
    """
    run the mcmc algorithm to do the sampling for a given length with given starting value
    """
    start = datetime.datetime.now()
    output = []
    for j in range(int(run_length/10000)+1):
        print("C size: " + str(size) + ", replication '0000s:", j, " time: ", datetime.datetime.now() - start)
        inner_out = []
        for i in range(10000):
            theta = theta_next_c(theta, candidate_std, size)
            inner_out.append(theta)
        output = output + inner_out
    np.save("mcmc_out/out_c_online_" + string + ".npy", output[-run_length:])


def mcmc_p(run_length, theta, string, candidate_std, size):
    """
    run the mcmc algorithm to do the sampling for a given length with given starting value
    """
    start = datetime.datetime.now()
    output = []
    for j in range(int(run_length/10000)+1):
        print("P size: " + str(size) + ", replication '0000s:", j, " time: ", datetime.datetime.now() - start)
        inner_out = []
        for i in range(10000):
            theta = theta_next_p(theta, candidate_std, size)
            inner_out.append(theta)
        output = output + inner_out
    np.save("mcmc_out/out_p_online_" + string + ".npy", output[-run_length:])


def run_both(run_length, theta, string, candidate_std, size, choice):
    if choice == "c":
        mcmc_c(run_length, theta, string, candidate_std, size)
    elif choice == "p":
        mcmc_p(run_length, theta, string, candidate_std, size)
    else:
        return 0


def main_run(size=10, total=10):
    global samples_c, samples_p
    candidate_std = 0.025
    std_diff = 0.005
    start = datetime.datetime.now()
    length = 100000
    t_start = 0.075
    data_size = size * 2 ** (total - 1)
    samples_c = np.random.exponential(1/lam_c, data_size)
    samples_p = np.random.exponential(1/lam_p, data_size)
    input_data = {"size": data_size, "cust": samples_c, "prov": samples_p}
    np.save("input_data/input_data_online.npy", input_data)
    arg_list = []
    for i in range(total):
        args = (length, t_start, str(i), candidate_std - int(i/2) * std_diff, size * 2 ** i, "c")
        arg_list.append(args)
        args = (length, t_start, str(i), candidate_std - int(i/2) * std_diff, size * 2 ** i, "p")
        arg_list.append(args)
    pool = Pool()
    res = pool.starmap(run_both, arg_list)
    pool.close()
    pool.join()
    end = datetime.datetime.now()
    print("done! time: ", end-start)


if __name__ == '__main__':
    main_run()
