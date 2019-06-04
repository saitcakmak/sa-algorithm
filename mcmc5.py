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
    0.10973439, 0.02929265, 0.03614674, 0.09902087, 0.04123991,
    0.02650089, 0.00162431, 0.08406197, 0.01614598, 0.08460564,
    0.01597884, 0.11282231, 0.00736363, 0.04953269, 0.02843456,
    0.00281063, 0.00324164, 0.00225956, 0.01468764, 0.01575224,
    0.00866736, 0.01886721, 0.06589302, 0.00083908, 0.02607207,
    0.07144814, 0.14248836, 0.00652778, 0.03983425, 0.03469301,
    0.02147575, 0.11583347, 0.02359936, 0.00066698, 0.02728315,
    0.05313919, 0.02034879, 0.01705029, 0.02318133, 0.01036422,
    0.0404005 , 0.03185449, 0.01162786, 0.02565473, 0.04138977,
    0.00781324, 0.02609438, 0.03374062, 0.02389158, 0.06040434,
    0.16189803, 0.0101373 , 0.02476581, 0.03506686, 0.03747814,
    0.0363562 , 0.017296  , 0.00599187, 0.00608198, 0.06378415,
    0.05288298, 0.01873092, 0.02996837, 0.02495186, 0.01220956,
    0.06557476, 0.0609865 , 0.05575857, 0.02582346, 0.0483248 ,
    0.09634731, 0.0947167 , 0.08331928, 0.1219832 , 0.11711338,
    0.0415837 , 0.0643063 , 0.01871938, 0.01053508, 0.08903397,
    0.2122535 , 0.0515617 , 0.00940745, 0.09897237, 0.10229418
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
    0.10408195, 0.17130904, 0.18556129, 0.00399131, 0.69711578,
    0.04790141, 0.15164982, 1.42477636, 0.02944946, 0.12286143,
    0.11373831, 0.16972504, 0.57821011, 0.32482889, 0.8022442 ,
    0.43090942, 0.03942842, 0.07116706, 0.0870168 , 0.00838413,
    0.45071738, 0.73981959, 0.03413619, 0.04891808, 0.24898838,
    0.10516123, 0.17259603, 0.01273279, 0.4066477 , 0.11376159,
    0.75553707, 0.16556259, 0.31601565, 0.03837743, 0.29015618,
    0.02814161, 0.03181805, 0.04125497, 0.0328924 , 0.00416528,
    0.00730777, 0.0086971 , 0.06868034, 0.25934587, 0.1238275 ,
    0.16193517, 0.17111358, 0.44066603, 0.11257094, 0.09068547,
    0.07075993, 0.49111528, 0.03541884, 0.03650929, 0.09415016,
    0.4554351 , 0.2664517 , 0.26036939, 0.22774755, 0.03749608,
    1.20578707, 0.32849224, 0.06086745, 0.04507833, 0.27966653,
    0.29344745, 0.04307139, 0.10568805, 0.2086434 , 0.17741725,
    0.22050728, 0.27857692, 0.02674967, 0.09055508, 0.73222198,
    0.0419883 , 0.07614342, 0.11949703, 0.0022883 , 0.22786195,
    0.16246907, 0.28929391, 0.07348032, 0.83085304, 0.01663301
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
