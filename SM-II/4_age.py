# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 10:00:20 2020.

@author: 10259

Experiment on age.
"""

import numpy as np
import time
from softmax import configuration, workload, matrix_query, func_var
from softmax import gm0_variance
from convexdp import ConvexDP, ca_variance, wCA, wCA2


def age():
    """Dataset AGE."""
    # Range queires for each gender
    first_1 = np.eye(2)
    second_1 = workload(116)
    work_gender = np.kron(first_1, second_1)

    # Range queires for both
    first_2 = np.ones((1, 2))
    second_2 = workload(116)
    work_both = np.kron(first_2, second_2)

    # identity querry
    eye = np.eye(232)
    work_all = np.concatenate((work_gender, work_both, eye))

    var = np.ones(116*5)
    # var[-232:] = 5
    var[0] = 1
    var[116] = 1

    # add_1 = np.zeros((1, 232))
    # add_1[18:116] = 1
    # add_2 = np.zeros((1, 232))
    # add_2[134:232] = 1
    # add_3 = add_1 + add_2

    # work_all = np.concatenate((work_gender, work_age, add_1, add_2, add_3))

    # var = np.ones(116*3+3)*10
    # var[98] = 1
    # var[214] = 1
    # var[330] = 1

    return work_all, var


if __name__ == '__main__':
    start = time.time()
    np.random.seed(0)
    work, bound = age()
    param_m, param_n = np.shape(work)

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

    wstrategy, wvar = wCA(work, bound, pcost)
    print("wvar=", np.max(wvar/bound))

    wstrategy2, wvar2 = wCA2(work, bound, pcost)
    print("wvar2=", np.max(wvar2/bound))

    gm0 = gm0_variance(work, pcost)
    print("gm0=", np.max(gm0/bound))

    end = time.time()
    print("time: ", end-start)
