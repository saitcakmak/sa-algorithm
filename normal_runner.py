import numpy as np
from naive_estimator import estimator
import datetime
from scipy.optimize import minimize
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Standardize
from botorch.utils import draw_sobol_samples
from typing import List
from multiprocessing import Pool

eps_num = 0.5
eps_base = 10
call_count = 0
n_step = 0


def analytic_value_VaR(x):
    """
    Calculates the true value of the BRO objective
    :param x:
    :param rho:
    :param alpha:
    :return:
    this is for VaR 0.75
    for other values change z as described in the paper.
    minimizer: x = 0.586924
    """
    mu_H = -15 * x + 10 * x ** 2
    z = 0.67448975
    sigma_H = np.sqrt((15 ** 2 + 16) * x ** 2 - 300 * x ** 3 + (10 ** 2 + 4) * x ** 4)
    return mu_H + z * sigma_H


def estimate(x, n, alpha, rho, mu_1, mu_2, sigma_1, sigma_2, seed=None):
    try:
        float(x)
    except TypeError:
        raise ValueError('This works with only 1 x at a time. Use a for loop instead.')
    if seed is not None:
        old_state = np.random.get_state()
        np.random.seed(seed)
    if rho != "mle":
        t_list = np.random.normal(0, 1, (n, 2))
        t_list[:, 0] = t_list[:, 0] * sigma_1 + mu_1
        t_list[:, 1] = t_list[:, 1] * sigma_2 + mu_2
    else:
        t_0 = np.ones(n, 1) * mu_1
        t_1 = np.ones(n, 1) * mu_2
        t_list = np.concatenate((t_0, t_1), axis=1)
    m = int(n / 5)
    val, der = estimator(t_list, float(x), m, alpha, rho, "normal")
    if seed is not None:
        np.random.set_state(old_state)
    return val, der


def estimate_no_grad(x, n0, *args):
    """
    Enables function evaluation without gradient.
    For use with a gradient-free optimization method.
    :param args:
    :param kwargs:
    :return:
    """
    global call_count
    n = n0 + int(n_step * call_count)
    val, der = estimate(x, n, *args)
    call_count += 1
    return val


def no_grad_multi_eval(x, *args):
    """
    Calls estimate_no_grad in a for loop.
    :param x:
    :param args:
    :return:
    """
    res = np.empty_like(x)
    for i in range(len(x)):
        res[i] = estimate_no_grad(x[i], *args)
    return res


def estimate_w_grad(x, n0, *args):
    """
    Enables function evaluation with  gradient.
    For use with a gradient based optimization method in scipy.minimize
    :param args:
    :param kwargs:
    :return:
    """
    global call_count
    n = n0 + int(n_step * call_count)
    val, der = estimate(x, n, *args)
    call_count += 1
    return np.array([val]), np.array([der])


def SA_run(seed, alpha, rho, x0=5, n0=100, iter_count=1000, mu_1=2, mu_2=5, sigma_1=1, sigma_2=1, SAA_seed=None):
    """
    Does a single run of the SA algorithm for the simple normal problem.
    :param seed: random seed
    :param alpha: risk level
    :param rho: risk measure
    :param x0: starting solution
    :param n0: outer sample starting size
    :param iter_count: number of iterations
    :param kwargs: passed to estimator
    :param SAA_seed: if given, an SAA version is run with this seed.
    :return:
    """
    np.random.seed(seed)
    begin = datetime.datetime.now()
    val_list = []
    der_list = []
    x_list = [x0]
    for t in range(1, iter_count + 1):
        eps = eps_num / (eps_base + t) ** 0.8
        n = n0 + int(n_step * t)
        val, der = estimate(x_list[t - 1], n, alpha, rho, mu_1, mu_2, sigma_1, sigma_2, SAA_seed)
        x_next = np.array(x_list[t - 1]) - eps * np.array(der)
        x_list.append(x_next)
        val_list.append(val)
        der_list.append(der)
        now = datetime.datetime.now()
        print(rho + "_" + str(alpha) + " t = ", t, " x = ", x_list[t], " val = ", val, " der = ",
              der, " time: ", now - begin)
    # np.save("sa_out/normal/" + rho + "_" + str(alpha) + "_iter_" + str(iter_count) + "_eps" + str(
        # eps_num) + "-" + str(eps_base) + "_x.npy", x_list)
    return x_list[1:]


def NM_run(seed, alpha, rho, x0=5, n0=100, iter_count=1000, mu_1=2, mu_2=5, sigma_1=1, sigma_2=1, SAA_seed=None):
    """
    Does a single run of the scipy.minimize 'Nelder-Mead' for the simple normal problem, without derivatives
    :param seed: random seed
    :param alpha: risk level
    :param rho: risk measure
    :param x0: starting solution
    :param n0: outer sample starting size
    :param iter_count: number of iterations
    :param kwargs: passed to estimator
    :param SAA_seed: if given, an SAA version is run with this seed.
    :return:
    """
    np.random.seed(seed)
    begin = datetime.datetime.now()
    res = minimize(estimate_no_grad, np.array([x0]),
                   args=(n0, alpha, rho, mu_1, mu_2, sigma_1, sigma_2, SAA_seed),
                   method='Nelder-Mead',
                   options={'disp': True,
                            'maxiter': iter_count,
                            'maxfev': iter_count,
                            'return_all': True})
    print(res)
    x_list = np.array(res.allvecs).reshape(-1).tolist()
    now = datetime.datetime.now()
    print('done time: %s' % (now - begin))
    print('call count: %d' % call_count)
    # np.save("sa_out/normal/NM_" + rho + "_" + str(alpha) + "_iter_" + str(iter_count) + "_x.npy", x_list)
    return x_list


def LBFGS_run(seed, alpha, rho, x0=5, n0=100, iter_count=1000, mu_1=2, mu_2=5, sigma_1=1, sigma_2=1, SAA_seed=None):
    """
    Does a single run of the scipy.minimize 'Nelder-Mead' for the simple normal problem, without derivatives
    :param seed: random seed
    :param alpha: risk level
    :param rho: risk measure
    :param x0: starting solution
    :param n0: outer sample starting size
    :param iter_count: number of iterations
    :param kwargs: passed to estimator
    :param SAA_seed: if given, an SAA version is run with this seed.
    :return:
    """
    np.random.seed(seed)
    begin = datetime.datetime.now()
    res = minimize(estimate_w_grad, np.array([x0]),
                   args=(n0, alpha, rho, mu_1, mu_2, sigma_1, sigma_2, SAA_seed),
                   method='L-BFGS-B',
                   jac=True,
                   bounds=[(-10, 10)],
                   options={'disp': True,
                            'maxiter': iter_count,
                            'maxfun': iter_count,
                            # 'return_all': True
                            }
                   )
    print(res)
    x_list = res.x.tolist()
    now = datetime.datetime.now()
    print('done time: %s' % (now - begin))
    print('call count: %d' % call_count)
    # np.save("sa_out/normal/BFGS_" + rho + "_" + str(alpha) + "_iter_" + str(iter_count) + "_x.npy", x_list)
    return x_list


def EI_run(seed, alpha, rho, x0=5, n0=100, iter_count=1000, mu_1=2, mu_2=5, sigma_1=1, sigma_2=1, SAA_seed=None):
    """
    Does a single run of the Expected Improvement algorithm for the simple normal problem, without derivatives
    :param seed: random seed
    :param alpha: risk level
    :param rho: risk measure
    :param x0: Ignored! Just to keep the same arglist as others
    :param n0: outer sample starting size
    :param iter_count: number of iterations
    :param kwargs: passed to estimator
    :param SAA_seed: if given, an SAA version is run with this seed.
    :return:
    """
    np.random.seed(seed)
    begin = datetime.datetime.now()
    args = (n0, alpha, rho, mu_1, mu_2, sigma_1, sigma_2, SAA_seed)

    points = torch.empty(iter_count, 1)
    values = torch.empty(points.shape)
    points[:4] = draw_sobol_samples(torch.tensor([[-5.], [5.]]), n=4, q=1).reshape(-1, 1)
    for i in range(4):
        values[i] = estimate_no_grad(points[i], *args)

    for i in range(4, iter_count):
        # fit gp
        # this transforms the GP to unit domain - botorch priors work best there
        transformed_points = points / 10. + 0.5
        model = SingleTaskGP(transformed_points[:i], values[:i], outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        # optimize EI to get the candidate
        acqf = ExpectedImprovement(model, best_f=torch.min(values), maximize=False)
        best_p, _ = optimize_acqf(acqf, bounds=torch.tensor([[0.], [1.]]), q=1, num_restarts=10, raw_samples=50)
        # transform it back to original domain
        best_p = best_p.detach() * 10. - 5.
        points[i] = best_p
        values[i] = estimate_no_grad(points[i], *args)

    best_list = torch.empty(points.shape)
    for i in range(1, iter_count+1):
        # pick the arg min of the history to return
        best_ind = torch.argmin(values[:i], dim=0)
        best_list[i-1] = points[best_ind]

    x_list = best_list
    now = datetime.datetime.now()
    print('done time: %s' % (now - begin))
    print('call count: %d' % call_count)
    # np.save("sa_out/normal/EI_" + rho + "_" + str(alpha) + "_iter_" + str(iter_count) + "_x.npy", x_list)
    return x_list


def evaluate(out_dict, n):
    """
    evaluates the data provided by the out_dict. Evaluates these solutions using the true objective
    set only for VaR 0.75
    :param out_dict: out data dict
    :param n: just used for output name
    :return: plot
    """
    out = dict()
    for key, entry in out_dict.items():
        out[key] = dict()
        for it_count, data in entry.items():
            total = 0.
            count = 0
            for x_list in data.values():
                total += analytic_value_VaR(x_list[-1])
                count += 1
            out[key][it_count] = total / count
    np.save('normal_out_all_%d.npy' % n, out)
    print(out)


def multi_run(replications: int, iters: List, n: int):
    """
    Runs all the algorithms defined above for the given number of replications
    using the kwargs defined below.
    The algorithms are set up so that they use at most the given number of function evaluations
    :param replications:
    :param iters: A list of number of iterations to run the algorithms for
    :param n: budget n
    :return:
    """
    global call_count
    kwargs = {
        'alpha': 0.75,
        'rho': 'VaR',
        'x0': 5,
        'n0': n,
        'mu_1': -15,
        'mu_2': 10,
        'sigma_1': 16,
        'sigma_2': 4
    }

    out_dict = {
        'SA': dict(),
        'SA_SAA': dict(),
        'NM': dict(),
        'NM_SAA': dict(),
        'LBFGS': dict(),
        'LBFGS_SAA': dict(),
        'EI': dict(),
        'EI_SAA': dict()
    }
    total_calls = dict()
    for key in out_dict.keys():
        total_calls[key] = dict()
    for it_count in iters:
        kwargs['iter_count'] = it_count
        for key in out_dict.keys():
            out_dict[key][it_count] = dict()
            total_calls[key][it_count] = 0
        i = 0
        while i < replications:
            try:
                out_dict['SA'][it_count][i] = SA_run(seed=i, **kwargs)
                total_calls['SA'][it_count] += call_count
                call_count = 0
                out_dict['SA_SAA'][it_count][i] = SA_run(seed=i, **kwargs, SAA_seed=i)
                total_calls['SA_SAA'][it_count] += call_count
                call_count = 0
                out_dict['NM'][it_count][i] = NM_run(seed=i, **kwargs)
                total_calls['NM'][it_count] += call_count
                call_count = 0
                out_dict['NM_SAA'][it_count][i] = NM_run(seed=i, **kwargs, SAA_seed=i)
                total_calls['NM_SAA'][it_count] += call_count
                call_count = 0
                out_dict['LBFGS'][it_count][i] = LBFGS_run(seed=i, **kwargs)
                total_calls['LBFGS'][it_count] += call_count
                call_count = 0
                out_dict['LBFGS_SAA'][it_count][i] = LBFGS_run(seed=i, **kwargs, SAA_seed=i)
                total_calls['LBFGS_SAA'][it_count] += call_count
                call_count = 0
                out_dict['EI'][it_count][i] = EI_run(seed=i, **kwargs)
                total_calls['EI'][it_count] += call_count
                call_count = 0
                out_dict['EI_SAA'][it_count][i] = EI_run(seed=i, **kwargs, SAA_seed=i)
                total_calls['EI_SAA'][it_count] += call_count
                call_count = 0
                i += 1
            except:
                continue
    np.save('call_counts_%d.npy' % n, total_calls)
    evaluate(out_dict, n)


if __name__ == "__main__":
    replications = 50
    iters = [10, 20, 50, 100]
    multi_run(replications, iters, n=100)
    multi_run(replications, iters, n=400)
    multi_run(replications, iters, n=1000)



