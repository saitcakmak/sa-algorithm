import numpy as np
import datetime


start = datetime.datetime.now()
candidate_std = 0.005
delta = 10 ** -6

K_c = 40
K_p = 20
price = 10
theta_c = 0.1
theta_p = 0.05

lam_c = K_c * 2 * np.exp(- theta_c * price) / (1 + np.exp(- theta_c * price))
lam_p = K_p * (1 - np.exp(- theta_p * price)) / (1 + np.exp(- theta_p * price))

lb = 0
ub = 0.5


def lr_c(candidate, theta):
    ratio_list = []
    for entry in samples_c:
        lam_cand = K_c * 2 * np.exp(- candidate * price) / (1 + np.exp(- candidate * price))
        lam_curr = K_c * 2 * np.exp(- theta * price) / (1 + np.exp(- theta * price))
        p_cand = np.exp(- lam_cand * (entry - delta)) - np.exp(- lam_cand * entry)
        p_curr = np.exp(- lam_curr * (entry - delta)) - np.exp(- lam_curr * entry)
        ratio_list.append(p_cand/p_curr)
    prob = np.prod(ratio_list)
    return np.nan_to_num(prob)


def lr_p(candidate, theta):
    ratio_list = []
    for entry in samples_p:
        lam_cand = K_p * (1 - np.exp(- candidate * price)) / (1 + np.exp(- candidate * price))
        lam_curr = K_p * (1 - np.exp(- theta * price)) / (1 + np.exp(- theta * price))
        p_cand = np.exp(- lam_cand * (entry - delta)) - np.exp(- lam_cand * entry)
        p_curr = np.exp(- lam_curr * (entry - delta)) - np.exp(- lam_curr * entry)
        ratio_list.append(p_cand/p_curr)
    prob = np.prod(ratio_list)
    return np.nan_to_num(prob)


def theta_next_c(theta):
    """
    return the next theta according to the accept/reject decision
    """
    candidate = np.random.normal(theta, candidate_std)
    accept_prob = lr_c(candidate, theta)
    uniform = np.random.random_sample()
    if uniform < accept_prob and ub > candidate > lb:
        return candidate
    else:
        return theta


def theta_next_p(theta):
    """
    return the next theta according to the accept/reject decision
    """
    candidate = np.random.normal(theta, candidate_std)
    accept_prob = lr_p(candidate, theta)
    uniform = np.random.random_sample()
    if uniform < accept_prob and ub > candidate > lb:
        return candidate
    else:
        return theta


def mcmc_c(run_length, theta, string):
    """
    run the mcmc algorithm to do the sampling for a given length with given starting value
    """
    output = []
    for j in range(int(run_length/10000)+1):
        print("C replication '0000s:", j, " time: ", datetime.datetime.now() - start)
        inner_out = []
        for i in range(10000):
            theta = theta_next_c(theta)
            inner_out.append(theta)
        output = output + inner_out
    np.save("mcmc_out/out_c_" + string + ".npy", output[-run_length:])


def mcmc_p(run_length, theta, string):
    """
    run the mcmc algorithm to do the sampling for a given length with given starting value
    """
    output = []
    for j in range(int(run_length/10000)+1):
        print("P replication '0000s:", j, " time: ", datetime.datetime.now() - start)
        inner_out = []
        for i in range(10000):
            theta = theta_next_p(theta)
            inner_out.append(theta)
        output = output + inner_out
    np.save("mcmc_out/out_p_" + string + ".npy", output[-run_length:])


if __name__ == "__main__":
    size = 10
    out_string = input("output string: ")
    length = 100000
    t_start = 0.075
    samples_c = np.random.exponential(lam_c, size)
    samples_p = np.random.exponential(lam_p, size)
    input_data = {"size": size, "cust": samples_c, "prov": samples_p}
    np.save("input_data_" + out_string + ".npy", input_data)
    mcmc_c(length, t_start, out_string)
    mcmc_p(length, t_start, out_string)
    end = datetime.datetime.now()
    print("done! time: ", end-start)
