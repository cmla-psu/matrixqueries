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
from softmax import configuration, workload, matrix_query, func_var
from convexdp import ConvexDP, ca_variance, wCA, wCA2


def WRange(param_m, param_n, uniform=False):
    """Range Workload of size (m, n)."""
    work = np.zeros([param_m, param_n])
    var = np.ones(param_m)
    for i in range(param_m):
        num_1 = np.random.randint(param_n)
        num_2 = np.random.randint(param_n)
        num = np.random.randint(1, 11)
        num = 1
        if num_1 < num_2:
            low = num_1
            high = num_2
        else:
            low = num_2
            high = num_1
        work[i, low:high+1] = 1
        var[i] = num
    if uniform:
        var = np.ones(param_m)
    return work, var


if __name__ == '__main__':
    start = time.time()
    np.random.seed(0)
    param_n = 64
    param_m = param_n * 2
    # set uniform=True to use uniform accuracy constraints
    # set uniform=False to use random accuracy constraints
    work, bound = WRange(param_m, param_n, uniform=True)
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
    # scale the covariance matrix
    # make it satisfy all accuracy constraints
    mat_cov = mat_opt.cov/np.max(mat_opt.f_var)

    acc = func_var(mat_cov, index)
    print("SM-II=", np.max(acc/bound))
    print("IP=", np.max(mat_opt.gm/bound))
    print("HM=", np.max(mat_opt.hm/bound))

    # run CA algorithm
    strategy = ConvexDP(work)
    pcost = mat_opt.pcost
    var = ca_variance(work, strategy, pcost)
    print("CA=", np.max(var/bound))

    wstrategy, wvar = wCA(work, bound, pcost)
    print("wCA-I=", np.max(wvar/bound))

    wstrategy2, wvar2 = wCA2(work, bound, pcost)
    print("wCA-II=", np.max(wvar2/bound))

    end = time.time()
    print("time: ", end-start)
