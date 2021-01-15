# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 10:00:20 2020.

@author: 10259

Experiment on range queries.
"""

import numpy as np
import time
from softmax import configuration, workload, matrix_query, func_var
from convexdp import ConvexDP, ca_variance


def WRange(param_m, param_n):
    """Range Workload."""
    work = np.zeros([param_m, param_n])
    var = np.ones(param_m)
    for i in range(param_m):
        num_1 = np.random.randint(param_n)
        num_2 = np.random.randint(param_n)
        num = np.random.randint(1, 11)
        if num_1 < num_2:
            low = num_1
            high = num_2
        else:
            low = num_2
            high = num_1
        work[i, low:high+1] = 1
        var[i] = num
    return work, var


if __name__ == '__main__':
    start = time.time()
    np.random.seed(0)
    param_n = 4
    param_m = 2*param_n
    param_s = param_n // 2
    work, bound = WRange(param_m, param_n)

    # configuration parameters
    args = configuration()
    args.init_mat = 'id_index'
    args.basis = 'work'
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

    end = time.time()
    print("time: ", end-start)
