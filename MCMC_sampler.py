import numpy as np
import datetime


start = datetime.datetime.now()
N = 40
prices = [5, 6, 7, 8, 10, 12, 15]
prior_mu = [np.log(22), np.log(8), np.log(18)]
prior_sigma = [2, 1, 2]
candidate_cov = np.diag([1, 0.25, 1])
delta = 0.000001
theta_c = [25, 10, 15]


def sample_from_true(N, price_list):
    """
    sample the data set of size N from the true distribution
    """
    global theta_c
    interarrival_cust = []
    interarrival_serv = []
    for p in price_list:
        lam = theta_c[0] * np.exp(- p / theta_c[1])
        mu = theta_c[2] * np.log(p)
        cust_seed = np.random.random(N)
        serv_seed = np.random.random(N)
        cust_log = np.log(cust_seed)
        serv_log = np.log(serv_seed)
        inner_cust = (-1 / lam) * cust_log
        inner_serv = (-1 / mu) * serv_log
        interarrival_cust.append((p, inner_cust))
        interarrival_serv.append((p, inner_serv))
    return [interarrival_cust, interarrival_serv]


data = sample_from_true(N, prices)


def prior_pdf(theta):
    """
    return the prior probability for a given theta
    let's use log-normal priors
    """
    global prior_mu, prior_sigma
    p_0 = (1 / (theta[0] * prior_sigma[0] * np.sqrt(2 * np.pi))) * np.exp(- (np.log(theta[0] - prior_mu[0])) ** 2 / (2 * prior_sigma[0] ** 2))
    p_1 = (1 / (theta[1] * prior_sigma[1] * np.sqrt(2 * np.pi))) * np.exp(- (np.log(theta[1] - prior_mu[1])) ** 2 / (2 * prior_sigma[1] ** 2))
    p_2 = (1 / (theta[2] * prior_sigma[2] * np.sqrt(2 * np.pi))) * np.exp(- (np.log(theta[2] - prior_mu[2])) ** 2 / (2 * prior_sigma[2] ** 2))
    return np.nan_to_num(p_0 * p_1 * p_2)


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
            lam_cand = candidate[0] * np.exp(- entry[0] / candidate[1])
            lam_curr = theta[0] * np.exp(- entry[0] / theta[1])
            p_cand = np.exp(- lam_cand * (val - delta)) - np.exp(- lam_cand * val)
            p_curr = np.exp(- lam_curr * (val - delta)) - np.exp(- lam_curr * val)
            ratio_list.append(p_cand/p_curr)
    for entry in interarrival_serv:
        for val in entry[1]:
            lam_cand = candidate[2] * np.log(entry[0])
            lam_curr = theta[2] * np.log(entry[0])
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
    prior_ratio = prior_pdf(candidate) / (prior_pdf(theta))
    # prior_ratio = 1
    accept_prob = prior_ratio * likelihood_ratio(candidate, theta)
    uniform = np.random.random_sample()
    if uniform < accept_prob:
        return candidate
    else:
        return theta


def mcmc_run(run_length, theta):
    """
    run the mcmc algorithm to do the sampling for a given length with given starting value
    """
    output = []
    for j in range(int(run_length/1000)):
        print("replication '000s:", j, " time: ", datetime.datetime.now() - start)
        for i in range(1000):
            theta = theta_next(theta)
            output.append(theta)
        np.save("temp_out.npy", output)
    return output


if __name__ == "__main__":
    length = 10000000
    t_start = [10, 10, 10]
    run = mcmc_run(length, t_start)
    params = {"len": length, "t_start": t_start, "N": N, "prices": prices, "prior_mu": prior_mu,
              "prior_sigma": prior_sigma, "covariance": candidate_cov, "delta": delta, "t_correct": theta_c}
    out = {"run": run, "params": params}
    np.save("mcmc_len_" + str(length) + "_.npy", out)
    end = datetime.datetime.now()
    print("done! time: ", end-start)
