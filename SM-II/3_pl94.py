# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 10:00:20 2020.

@author: 10259

Experiment on discrete queries.
"""

import numpy as np
import time
from softmax import configuration, workload, matrix_query, func_var
from convexdp import ConvexDP, ca_variance


def PL94():
    """Dataset PL94."""
    first_1 = np.eye(2)
    second_1 = np.ones((1, 2))
    third_1 = np.ones((1, 63))
    work_age = np.kron(np.kron(first_1, second_1), third_1)

    first_2 = np.ones((1, 2))
    second_2 = np.eye(2)
    third_2 = np.ones((1, 63))
    work_eth = np.kron(np.kron(first_2, second_2), third_2)

    first_3 = np.ones((1, 2))
    second_3 = np.ones((1, 2))
    third_3 = np.eye(63)
    work_race = np.kron(np.kron(first_3, second_3), third_3)

    work_id = np.eye(252)

    work_all = np.concatenate((work_age, work_eth, work_race, work_id))

    var = np.ones(319)
    var[10: 67] = 2
    var[67:] = 4

    return work_all, var


if __name__ == '__main__':
    start = time.time()
    np.random.seed(0)
    work, bound = PL94()
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

    end = time.time()
    print("time: ", end-start)
