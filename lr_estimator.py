import numpy as np
import problem_sampler
import lr_calculator
import scipy.stats as sci

# Confidence of the t-test
conf = 0.95


def estimator(theta_list, x, m, alpha, rho, prob):
    if prob == "simple":
        sampler = problem_sampler.simple_sampler_lr
        lr_calc = lr_calculator.simple_lr
        dim = 2
    elif prob == "two_sided":
        sampler = problem_sampler.two_sided_sampler_lr
        lr_calc = lr_calculator.two_sided_lr
        dim = 200
    elif prob == "quad":
        sampler = problem_sampler.quad_sampler_lr
        lr_calc = lr_calculator.quad_lr
        dim = 1
    else:
        return -1
    n = len(theta_list)
    mean_est = np.zeros(n)
    std_est = np.zeros(n)
    vals = np.zeros((n, m))
    ders = np.zeros((n, m))
    rvs = np.zeros((n, m, dim))
    likelihoods = np.zeros((n, m, dim))

    # Draw samples and keep track of the data and estimated means & std dev
    for i in range(n):
        samples = sampler(theta_list[i], x, m)
        vals[i] = samples[0]
        ders[i] = samples[1]
        rvs[i] = samples[2]
        likelihoods[i] = samples[3]
        mean = np.average(samples[0])
        std = np.std(samples[0])
        mean_est[i] = mean
        std_est[i] = std

    # Sort all according to the estimated means
    sort_index = np.argsort(mean_est)
    theta_list = np.array(theta_list)[sort_index]
    mean_est = mean_est[sort_index]
    std_est = std_est[sort_index]
    vals = vals[sort_index]
    ders = ders[sort_index]
    rvs = rvs[sort_index]
    likelihoods = likelihoods[sort_index]
    var_mean = mean_est[int(n * alpha)]
    var_std = std_est[int(n * alpha)]

    spare = 0
    updated_list = []
    updated_vals = []
    updated_ders = []
    updated_rvs = []
    updated_likelihoods = []

    # Select the list that goes into the LR calculations
    if rho == "VaR":
        for i in range(n):
            diff = abs(var_mean - mean_est[i])
            theta_std = std_est[i]
            std = np.sqrt(theta_std ** 2 / m + var_std ** 2 / m)
            if std:
                df = std ** 2 / ( (theta_std ** 2 / m) ** 2 + (var_std ** 2 / m) ** 2 ) * (m-1)
            else:
                df = 10
            if diff < std * sci.t.ppf(conf, df) or std == 0:
                updated_list.append(theta_list[i])
                updated_vals.append(vals[i])
                updated_ders.append(ders[i])
                updated_rvs.append(rvs[i])
                updated_likelihoods.append(likelihoods[i])
            elif mean_est[i] < var_mean:
                spare += 1

    elif rho == "CVaR":
        for i in range(n):
            if mean_est[i] > var_mean:
                updated_list.append(theta_list[i])
                updated_vals.append(vals[i])
                updated_ders.append(ders[i])
                updated_rvs.append(rvs[i])
                updated_likelihoods.append(likelihoods[i])
            else:
                diff = abs(var_mean - mean_est[i])
                theta_std = std_est[i]
                std = np.sqrt(theta_std ** 2 / m + var_std ** 2 / m)
                if std:
                    df = std ** 2 / ( (theta_std ** 2 / m) ** 2 + (var_std ** 2 / m) ** 2 ) * (m-1)
                else:
                    df = 10
                if diff < std * sci.t.ppf(conf, df) or std == 0:
                    updated_list.append(theta_list[i])
                    updated_vals.append(vals[i])
                    updated_ders.append(ders[i])
                    updated_rvs.append(rvs[i])
                    updated_likelihoods.append(likelihoods[i])
                else:
                    spare += 1

    # Calculate the LR estimates using the remaining data.
    lr_est = np.zeros(len(updated_list))
    lr_der_est = np.zeros(len(updated_list))

    for i in range(len(updated_list)):
        lr_vals = np.zeros((len(updated_list), m))
        lr_ders = np.zeros((len(updated_list), m))
        lr_weights = np.zeros((len(updated_list), m))
        for j in range(len(updated_list)):
            weights = lr_calc(updated_list[i], updated_rvs[j], updated_likelihoods[j], x)
            lr_vals[j] = updated_vals[j] * weights
            lr_ders[j] = updated_ders[j] * weights
            lr_weights[j] = weights
        total_weight = np.sum(lr_weights)
        lr_est[i] = np.sum(lr_vals) / total_weight
        lr_der_est[i] = np.sum(lr_ders) / total_weight

    # Sort and return the estimates.
    sort_index = np.argsort(lr_est)
    lr_est = lr_est[sort_index]
    lr_der_est = lr_der_est[sort_index]

    if rho == "VaR":
        return lr_est[int(n * alpha) - spare], lr_der_est[int(n * alpha) - spare]
    elif rho == "CVaR":
        return np.average(lr_est[int(n * alpha) - spare:]), np.average(lr_der_est[int(n * alpha) - spare:])
