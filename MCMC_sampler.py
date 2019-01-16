import numpy as np
import datetime


start = datetime.datetime.now()
candidate_cov = np.diag([0.16, 0.000001, 0.01, 0.00000025])
delta = 0.000001
theta_c = [40, 0.1, 10, 0.05]


def sample_from_true(N, price_list):
    """
    sample the data set of size M from the true distribution
    """
    global theta_c
    interarrival_cust = []
    interarrival_serv = []
    for p in price_list:
        lam = theta_c[0] * np.exp(- theta_c[1] * p) / (1 + np.exp(- theta_c[1] * p))
        mu = theta_c[2] / (1 + np.exp(- theta_c[3] * p))
        cust_seed = np.random.random(N)
        serv_seed = np.random.random(N)
        cust_log = np.log(cust_seed)
        serv_log = np.log(serv_seed)
        inner_cust = (-1 / lam) * cust_log
        inner_serv = (-1 / mu) * serv_log
        interarrival_cust.append((p, inner_cust))
        interarrival_serv.append((p, inner_serv))
    return [interarrival_cust, interarrival_serv]


def likelihood_ratio(candidate, theta):
    """
    calculate the likelihood of a given dataset for the given theta
    we should avoid constantly sending the dataset here
    """
    global data, delta
    interarrival_cust = data[0]
    interarrival_serv = data[1]
    ratio_list = []
    for entry in interarrival_cust:
        for val in entry[1]:
            lam_cand = candidate[0] * np.exp(- candidate[1] * entry[0]) / (1 + np.exp(- candidate[1] * entry[0]))
            lam_curr = theta[0] * np.exp(- theta[1] * entry[0]) / (1 + np.exp(- theta[1] * entry[0]))
            p_cand = np.exp(- lam_cand * (val - delta)) - np.exp(- lam_cand * val)
            p_curr = np.exp(- lam_curr * (val - delta)) - np.exp(- lam_curr * val)
            ratio_list.append(p_cand/p_curr)
    for entry in interarrival_serv:
        for val in entry[1]:
            lam_cand = candidate[2] / (1 + np.exp(- candidate[3] * entry[0]))
            lam_curr = theta[2] / (1 + np.exp(- theta[3] * entry[0]))
            p_cand = np.exp(- lam_cand * (val - delta)) - np.exp(- lam_cand * val)
            p_curr = np.exp(- lam_curr * (val - delta)) - np.exp(- lam_curr * val)
            ratio_list.append(p_cand/p_curr)
    prob = np.prod(ratio_list)
    return np.nan_to_num(prob)


def generate_candidate(theta):
    """
    generate the next candidate from normal distribution
    """
    global candidate_cov
    return np.random.multivariate_normal(theta, candidate_cov)


def theta_next(theta):
    """
    return the next theta according to the accept/reject decision
    """
    candidate = generate_candidate(theta)
    accept_prob = likelihood_ratio(candidate, theta)
    uniform = np.random.random_sample()
    if uniform < accept_prob:
        return candidate
    else:
        return theta


def mcmc_run(run_length, theta, string):
    """
    run the mcmc algorithm to do the sampling for a given length with given starting value
    """
    output = []
    for j in range(int(run_length/10000)):
        print("replication '0000s:", j, " time: ", datetime.datetime.now() - start)
        inner_out = []
        for i in range(10000):
            theta = theta_next(theta)
            inner_out.append(theta)
        output = output + inner_out
        np.save("output/out_" + string + ".npy", output)
    out_2 = output[int(len(output)/2):]
    np.save("output/out_" + string + ".npy", out_2)


def get_input():
    """
    ask user for N and prices
    """
    N = int(input("enter N: "))
    prices = []
    while True:
        p = float(input("enter prices (0 for exit): "))
        if p == 0:
            break
        else:
            prices.append(p)
    string = input("enter output string: ")
    return N, prices, string


if __name__ == "__main__":
    N, prices, string = get_input()
    length = 200000
    t_start = theta_c
    data = sample_from_true(N, prices)
    params = {"len": length, "t_start": t_start, "N": N, "prices": prices, "data": data,
              "covariance": candidate_cov, "delta": delta, "t_correct": theta_c}
    np.save("output/mcmc_params_" + string + ".npy", params)
    mcmc_run(length, t_start, string)
    end = datetime.datetime.now()
    print("done! time: ", end-start)
