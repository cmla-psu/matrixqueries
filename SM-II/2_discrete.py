# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 10:00:20 2020.

@author: 10259

Experiment on discrete queries.
"""

import numpy as np
import time
from softmax import configuration, workload, matrix_query, func_var
from softmax import gm0_variance
from convexdp import ConvexDP, ca_variance, wCA, wCA2


def WDiscrete(param_m, param_n, prob):
    """Discrete workload."""
    p = [1-prob, prob]
    work = np.random.choice(2, [param_m, param_n], p)
    # convert int to float, avoid strange bug
    work = 2.0*work - 1
    # size_1 = param_m // 10
    # var_1 = np.random.randint(1, 4, size=size_1)
    # size_2 = param_m - size_1
    # var_2 = np.random.randint(10, 16, size=size_2)
    # var = np.concatenate([var_1, var_2])
    var = np.random.randint(1, 11, param_m)
    # var = np.ones(param_m)
    return work, var


if __name__ == '__main__':
    start = time.time()
    np.random.seed(0)
    param_n = 1024
    param_m = 2*param_n
    # param_m = param_n // 2
    param_s = param_n // 2
    work, bound = WDiscrete(param_m, param_n, 0.2)

    # configuration parameters
    args = configuration()
    args.init_mat = 'id_index'
    # args.init_mat = 'hb'
    # args.init_mat = 'id'
    # args.id_factor = 20000
    # args.basis = 'work'
    args.basis = 'id'
    args.maxitercg = 5

    if args.basis == 'id':
        index = work
        basis = np.eye(param_n)
    if args.basis == 'work':
        basis = workload(param_n)
        index = work @ np.linalg.inv(basis)

    mat_opt = matrix_query(args, basis, index, bound)
    mat_opt.optimize()
    mat_cov = mat_opt.cov/np.max(mat_opt.f_var)

    acc = func_var(mat_cov, index)
    print("acc=", np.max(acc/bound), np.sum(acc))
    print("gm=", np.max(mat_opt.gm/bound), np.sum(mat_opt.gm))
    print("hm=", np.max(mat_opt.hm/bound), np.sum(mat_opt.hm))

    # run CA algorithm
    strategy = ConvexDP(work)
    pcost = mat_opt.pcost
    var = ca_variance(work, strategy, pcost)
    print("var=", np.max(var/bound), np.sum(var))

    wstrategy, wvar = wCA(work, bound, pcost)
    print("wvar=", np.max(wvar/bound), np.sum(wvar))

    wstrategy2, wvar2 = wCA2(work, bound, pcost)
    print("wvar2=", np.max(wvar2/bound), np.sum(wvar2))

    gm0 = gm0_variance(work, pcost)
    print("gm0=", np.max(gm0/bound), gm0*param_m)

    end = time.time()
    print("time: ", end-start)

    # variance_wca = 2.49
    variance_wca = np.max(var/bound)
    total_error = np.sum(var)
    ratio_gm = np.sum(mat_opt.gm)/total_error
    ratio_hm = np.sum(mat_opt.hm)/total_error
    ratio_wca1 = np.sum(wvar)/total_error
    ratio_wca2 = np.sum(wvar2)/total_error
    true_variance = total_error/np.max(var/bound)*variance_wca
    ratio_sm = np.sum(bound)/true_variance
    print('& {0:.2f} & {1:.2f} & {2:.2f} & {3:.2f} & {4:.2f}'.format(
        ratio_gm, ratio_hm, ratio_wca1, ratio_wca2, ratio_sm))
