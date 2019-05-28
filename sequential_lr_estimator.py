import numpy as np
import problem_sampler
import scipy.stats as sci
import lr_calculator

# Confidence for the t-test used and sequence of budget percentages
conf = 0.95
seq = np.array([0.12, 0.16, 0.22])
sampler = problem_sampler.simple_sampler
sampler_lr = problem_sampler.simple_sampler_lr
lr_calc = lr_calculator.simple_lr


def estimator(theta_list, x, m, alpha, rho):
    n = len(theta_list)
    total_budget = n * m
    budget_used = 0
    seq_budget = seq * total_budget
    updated_list = np.array(theta_list)
    spare = 0
    m_used = 0
    old_means = np.zeros(len(updated_list))
    old_std = np.zeros(len(updated_list))

    # Elimination stage
    for k in range(len(seq)):
        m = int(seq_budget[k] / len(updated_list))
        budget_used += len(updated_list) * m
        means = np.zeros(len(updated_list))
        std = np.zeros(len(updated_list))
        # Draw new samples for the survivors
        for i in range(len(updated_list)):
            inner_samples = sampler(updated_list[i], x, m)
            means[i] = (np.sum(inner_samples[0]) + old_means[i] * m_used) / (m + m_used)
            std[i] = np.sqrt((np.std(inner_samples[0]) ** 2 * m + old_std[i] ** 2 * m_used) / (m + m_used))

        sort_index = np.argsort(means)
        means = means[sort_index]
        std = std[sort_index]
        updated_list = updated_list[sort_index]

        var_mean = means[int(n * alpha) - spare]
        var_std = std[int(n * alpha) - spare]

        next_list = []
        old_means = []
        old_std = []
        m_used += m

        # Update the survivor list
        if rho == "VaR":
            for i in range(len(updated_list)):
                theta = updated_list[i]
                diff = abs(var_mean - means[i])
                theta_std = std[i]
                t_std = np.sqrt(theta_std ** 2 / m + var_std ** 2 / m)
                df = t_std ** 2 / ((theta_std ** 2 / m) ** 2 + (var_std ** 2 / m) ** 2) * (m - 1)
                if diff < t_std * sci.t.ppf(conf, df):
                    next_list.append(theta)
                    old_means.append(means[i])
                    old_std.append(std[i])
                elif means[i] < var_mean:
                    spare += 1

        elif rho == "CVaR":
            for i in range(len(updated_list)):
                theta = updated_list[i]
                if means[i] > var_mean:
                    next_list.append(theta)
                else:
                    diff = abs(var_mean - means[i])
                    theta_std = std[i]
                    t_std = np.sqrt(theta_std ** 2 / m + var_std ** 2 / m)
                    df = t_std ** 2 / ((theta_std ** 2 / m) ** 2 + (var_std ** 2 / m) ** 2) * (m - 1)
                    if diff < t_std * sci.t.ppf(conf, df):
                        next_list.append(theta)
                        old_means.append(means[i])
                        old_std.append(std[i])
                    else:
                        spare += 1

        updated_list = np.array(next_list)

    # Restart
    remaining_budget = total_budget - budget_used
    m = int(remaining_budget / len(updated_list))
    vals = np.zeros((len(updated_list), m))
    ders = np.zeros((len(updated_list), m))
    rvs = np.zeros((len(updated_list), m, problem_sampler.dim))
    likelihoods = np.zeros((len(updated_list), m, problem_sampler.dim))

    # New samples with the remaining budget and estimation
    for i in range(len(updated_list)):
        inner_samples = sampler_lr(updated_list[i], x, m)
        vals[i] = inner_samples[0]
        ders[i] = inner_samples[1]
        rvs[i] = inner_samples[2]
        likelihoods[i] = inner_samples[3]

    # Use the new samples to calculate LR estimators
    lr_est = np.zeros(len(updated_list))
    lr_der_est = np.zeros(len(updated_list))

    for i in range(len(updated_list)):
        lr_vals = np.zeros((len(updated_list), m))
        lr_ders = np.zeros((len(updated_list), m))
        lr_weights = np.zeros((len(updated_list), m))
        for j in range(len(updated_list)):
            weights = lr_calc(updated_list[i], rvs[j], likelihoods[j])
            lr_vals[j] = vals[j] * weights
            lr_ders[j] = ders[j] * weights
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