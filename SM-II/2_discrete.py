# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 10:00:20 2020.

@author: 10259

Experiment on discrete queries.
"""

import numpy as np
import time
from softmax import configuration, workload, matrix_query, func_var
from convexdp import ConvexDP, ca_variance, wCA


def WDiscrete(param_m, param_n, prob):
    """Discrete workload."""
    p = [1-prob, prob]
    work = np.random.choice(2, [param_m, param_n], p)
    # convert int to float, avoid strange bug
    work = 2.0*work - 1
    size_1 = param_m // 10
    var_1 = np.random.randint(1, 4, size=size_1)
    size_2 = param_m - size_1
    var_2 = np.random.randint(10, 16, size=size_2)
    var = np.concatenate([var_1, var_2])

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
    # args.id_factor = 16
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
    print("acc=", np.max(acc/bound))
    print("gm=", np.max(mat_opt.gm/bound))
    print("hm=", np.max(mat_opt.hm/bound))

    # run CA algorithm
    strategy = ConvexDP(work)
    pcost = mat_opt.pcost
    var = ca_variance(work, strategy, pcost)
    print("var=", np.max(var/bound))

    wstrategy, wvar = wCA(work, bound, pcost)

    end = time.time()
    print("time: ", end-start)
