import numpy as np
import datetime


start = datetime.datetime.now()
candidate_std = 0.005
delta = 10 ** -6

K_c = 40
K_p = 20

samples_c = [0.0195015, 0.02022016, 0.03236796, 0.00362713, 0.00183414,
             0.14411868, 0.10316238, 0.01894079, 0.00809587, 0.06989924,
             0.00774514, 0.11707596, 0.08575531, 0.07806506, 0.0167134,
             0.05983467, 0.10086276, 0.02607974, 0.0376224, 0.06864567
             ]
samples_p = [0.02718718, 0.33309152, 0.01486785, 0.47200697, 0.1829244,
             0.40767248, 0.24125914, 0.24619485, 0.06222773, 0.06732975,
             0.01778286, 0.01253464, 0.34979672, 0.11275023, 0.52416463,
             0.02169284, 0.11244249, 0.05019376, 0.02922926, 0.0904917
             ]

lb = 0
ub = 0.5

price = 10


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
    for j in range(int(run_length/10000)):
        print("C replication '0000s:", j, " time: ", datetime.datetime.now() - start)
        inner_out = []
        for i in range(10000):
            theta = theta_next_c(theta)
            inner_out.append(theta)
        output = output + inner_out
        np.save("mcmc_out/out_c_" + string + ".npy", output)


def mcmc_p(run_length, theta, string):
    """
    run the mcmc algorithm to do the sampling for a given length with given starting value
    """
    output = []
    for j in range(int(run_length/10000)):
        print("P replication '0000s:", j, " time: ", datetime.datetime.now() - start)
        inner_out = []
        for i in range(10000):
            theta = theta_next_p(theta)
            inner_out.append(theta)
        output = output + inner_out
        np.save("mcmc_out/out_p_" + string + ".npy", output)


if __name__ == "__main__":
    out_string = input("output string: ")
    length = 100000
    t_start = 0.075
    mcmc_c(length, t_start, out_string)
    mcmc_p(length, t_start, out_string)
    end = datetime.datetime.now()
    print("done! time: ", end-start)
