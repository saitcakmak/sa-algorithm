import numpy as np
import problem_sampler
import scipy.stats as sci

# Confidence for the t-test used and sequence of budget percentages
conf = 0.95
seq = np.array([0.12, 0.16, 0.22])


def estimator(theta_list, x, m, alpha, rho, prob):
    if prob == "simple":
        sampler = problem_sampler.simple_sampler
    elif prob == "two_sided":
        sampler = problem_sampler.two_sided_sampler
    else:
        return -1
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
                if t_std:
                    df = t_std ** 2 / ((theta_std ** 2 / m) ** 2 + (var_std ** 2 / m) ** 2) * (m - 1)
                else:
                    df = 10
                if diff < t_std * sci.t.ppf(conf, df) or t_std == 0:
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
                    old_means.append(means[i])
                    old_std.append(std[i])
                else:
                    diff = abs(var_mean - means[i])
                    theta_std = std[i]
                    t_std = np.sqrt(theta_std ** 2 / m + var_std ** 2 / m)
                    if t_std:
                        df = t_std ** 2 / ((theta_std ** 2 / m) ** 2 + (var_std ** 2 / m) ** 2) * (m - 1)
                    else:
                        df = 10
                    if diff < t_std * sci.t.ppf(conf, df) or t_std == 0:
                        next_list.append(theta)
                        old_means.append(means[i])
                        old_std.append(std[i])
                    else:
                        spare += 1

        updated_list = np.array(next_list)

    # Restart
    means = np.zeros(len(updated_list))
    ders = np.zeros(len(updated_list))

    remaining_budget = total_budget - budget_used
    m = int(remaining_budget / len(updated_list))

    # New samples with the remaining budget and estimation
    for i in range(len(updated_list)):
        inner_samples = sampler(updated_list[i], x, m)
        means[i] = np.average(inner_samples[0])
        ders[i] = np.average(inner_samples[1])

    sort_index = np.argsort(means)
    means = means[sort_index]
    ders = ders[sort_index]

    if rho == "VaR":
        return means[int(n * alpha) - spare], ders[int(n * alpha) - spare]
    elif rho == "CVaR":
        return np.average(means[int(n * alpha) - spare:]), np.average(ders[int(n * alpha) - spare:])
