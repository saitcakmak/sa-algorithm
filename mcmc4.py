import numpy as np
import datetime


start = datetime.datetime.now()
candidate_std = 0.005
delta = 10 ** -6

K_c = 40
K_p = 20

samples_c = [
    0.0195015, 0.02022016, 0.03236796, 0.00362713, 0.00183414,
    0.14411868, 0.10316238, 0.01894079, 0.00809587, 0.06989924,
    0.00774514, 0.11707596, 0.08575531, 0.07806506, 0.0167134,
    0.05983467, 0.10086276, 0.02607974, 0.0376224, 0.06864567,
    0.10331684, 0.01686447, 0.03518494, 0.06372492, 0.00813789,
    0.04800135, 0.07816179, 0.02912283, 0.01100758, 0.05733704,
    0.06946685, 0.0033816 , 0.11109014, 0.01415065, 0.02482517,
    0.02033625, 0.00570086, 0.09115745, 0.0401755 , 0.04345916,
    0.1320702 , 0.00718999, 0.01298182, 0.04139147, 0.00689293,
    0.03606104, 0.00921689, 0.00537867, 0.09623656, 0.05031075,
    0.0345942 , 0.06337521, 0.05935624, 0.09092151, 0.10512483,
    0.03067465, 0.01659172, 0.10176736, 0.09886942, 0.00580274,
    0.09678769, 0.0168746 , 0.09143957, 0.01796036, 0.04759073,
    0.06470269, 0.14635024, 0.03739571, 0.02283999, 0.01180724,
    0.03411887, 0.04891418, 0.05167688, 0.08381001, 0.00289396,
    0.10973439, 0.02929265, 0.03614674, 0.09902087, 0.04123991
    ]
samples_p = [
    0.02718718, 0.33309152, 0.01486785, 0.47200697, 0.1829244,
    0.40767248, 0.24125914, 0.24619485, 0.06222773, 0.06732975,
    0.01778286, 0.01253464, 0.34979672, 0.11275023, 0.52416463,
    0.02169284, 0.11244249, 0.05019376, 0.02922926, 0.0904917,
    0.16390829, 0.02779531, 0.18204641, 0.12718683, 0.25179972,
    0.16489294, 0.21366101, 0.10921757, 0.39965187, 0.17978787,
    0.73303824, 0.12269229, 0.05362327, 0.14212019, 0.06277188,
    0.09067714, 0.01503801, 0.37727311, 0.10173714, 0.08920901,
    0.00478748, 0.21452838, 0.05167996, 0.14929627, 0.10103538,
    0.1601979 , 0.0591951 , 0.11184328, 0.93935364, 0.22643587,
    0.11411803, 0.48565561, 0.050705  , 0.03996537, 0.07845103,
    0.18739405, 0.12948802, 0.15712169, 0.01091551, 0.20613724,
    0.06111084, 0.01613938, 0.04401627, 0.15374798, 0.30423083,
    0.1500902 , 0.20325979, 0.01289964, 0.36217272, 0.00876846,
    0.03024023, 0.10014791, 0.12609931, 0.04731845, 0.12666594,
    0.10408195, 0.17130904, 0.18556129, 0.00399131, 0.69711578
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
