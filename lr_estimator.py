import numpy as np
import problem_sampler
import lr_calculator
import scipy.stats as sci


def estimator(theta_list, x, m, alpha, rho):
    n = len(theta_list)
    data = {}
    mean_est = np.zeros(n)
    std_est = np.zeros(n)
    for i in range(n):
        samples = problem_sampler.sampler_lr(theta_list[i], x, m)
        data[theta]["samples"] = samples
        mean = np.average(samples[0])
        std = np.std(samples[0])
        mean_est[i] = mean
        std_est[i] = std

    sort_index = np.argsort(mean_est)
    theta_list = np.array(theta_list)[sort_index]
    mean_est = mean_est[sort_index]
    std_est = std_est[sort_index]
    var = theta_list[int(n * alpha)]
    var_mean = mean_est[int(n * alpha)]
    var_std = std_est[int(n * alpha)]

    spare = 0
    updated_list = []

    if rho == "VaR":
        for i in range(n):
            theta = theta_list[i]
            diff = abs(var_mean - mean_est[i])
            theta_std = std_est[i]
            std = np.sqrt(theta_std ** 2 / m + var_std ** 2 / m)
            df = std ** 2 / ( (theta_std ** 2 / m) ** 2 + (var_std ** 2 / m) ** 2 ) * (m-1)
            if diff < std * sci.t.ppf(0.95, df):
                updated_list.append(theta)
            elif mean_est[i] < var_mean:
                spare += 1

    elif rho == "CVaR":
        for i in range(n):
            theta = theta_list[i]
            if mean_est[i] > var_mean:
                updated_list.append(theta)
            else:
                diff = abs(var_mean - mean_est[i])
                theta_std = std_est[i]
                std = np.sqrt(theta_std ** 2 / m + var_std ** 2 / m)
                df = std ** 2 / ( (theta_std ** 2 / m) ** 2 + (var_std ** 2 / m) ** 2 ) * (m-1)
                if diff < std * sci.t.ppf(0.95, df):
                    updated_list.append(theta)
                else:
                    updated_list.append(theta)

    updated_est = np.zeros(len(updated_list))
    update_der = np.zeros(len(updated_list))
    # TODO: calculate the LR estimates for the updated list
    # TODO: this should follow along with lr_calculator. Otherwise, it will be too costly

    sort_index = np.argsort(updated_est)
    updated_list = np.array(updated_list)[sort_index]
    updated_est = updated_est[sort_index]
    update_der = update_der[sort_index]

    if rho == "VaR":
        return updated_est[int(n * alpha) - spare], update_der[int(n * alpha) - spare]
    elif rho == "CVaR":
        return np.average(updated_est[int(n * alpha) - spare:]), np.average(update_der[int(n * alpha) - spare:])
