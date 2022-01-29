# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2022 cmla-psu/Yingtai Xiao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import time
from softmax import configuration, workload, matrix_query, func_var, gm0_variance
from convexdp import ConvexDP, ca_variance, wCA, wCA2


def PL94():
    """Dataset PL94."""
    # age marginal
    first_1 = np.eye(2)
    second_1 = np.ones((1, 2))
    third_1 = np.ones((1, 63))
    work_age = np.kron(np.kron(first_1, second_1), third_1)

    # ethnicity marginal
    first_2 = np.ones((1, 2))
    second_2 = np.eye(2)
    third_2 = np.ones((1, 63))
    work_eth = np.kron(np.kron(first_2, second_2), third_2)

    # race marginal
    first_3 = np.ones((1, 2))
    second_3 = np.ones((1, 2))
    third_3 = np.eye(63)
    work_race = np.kron(np.kron(first_3, second_3), third_3)

    # identity queries
    work_id = np.eye(252)

    work_all = np.concatenate((work_age, work_eth, work_race, work_id))

    var = np.ones(319)
    var[10: 67] = 1
    var[67:] = 1

    # var[10: 67] = 2
    # var[67:] = 4

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
