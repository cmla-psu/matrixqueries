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


def kron_i(mat, vec, i):
    """Do kronecker productor i times, i >= 0."""
    res = mat
    for times in range(i):
        res = np.kron(res, vec)
    return res


def mat_i_j(d, mat, vec, i, j):
    """Two way margnial matrix."""
    res = 1
    for p_1 in range(i):
        res = np.kron(res, vec)
    res = np.kron(res, mat)
    for p_2 in range(j-i-1):
        res = np.kron(res, vec)
    res = np.kron(res, mat)
    for p_3 in range(d-j-1):
        res = np.kron(res, vec)
    return res


def marginal(d, k, choice, same):
    """
    k**d domain.

    choice: "one" -> one way marginal
            "two" -> two way marginal
    same: True -> all one variance bound
          False -> increasing variance bound
    """
    mat_id = np.eye(k)
    vec_one = np.ones((1, k))
    if choice == "one":
        work = kron_i(mat_id, vec_one, d-1)
        work_all = work
        for i in range(d-1):
            left = kron_i(vec_one, vec_one, i)
            tmp = np.kron(left, mat_id)
            work = kron_i(tmp, vec_one, d-2-i)
            work_all = np.concatenate((work_all, work))
        var = np.ones(d*k)
        if not same:
            for i in range(d):
                var[k*i: k*i+k] = i+1
    if choice == "two":
        size_m = k*k*d*(d-1)//2
        size_n = k**d
        work_all = np.zeros([size_m, size_n])
        var = np.ones(size_m)
        block = k*k
        s = 0
        for i in range(d-1):
            for j in range(i+1, d):
                work_all[block*s: block*s+block, :] = mat_i_j(
                    d, mat_id, vec_one, i, j)
                s = s+1
        if not same:
            for t in range(d*(d-1)//2):
                var[block*t: block*t+block] = t+1
    return work_all, var


if __name__ == '__main__':
    start = time.time()
    np.random.seed(0)
    # r is fixed, change t in [2, 16]
    r = 3
    t = 4
    work1, bound1 = marginal(r, t, "one", True)
    work2, bound2 = marginal(r, t, "two", True)
    work = np.concatenate((work1, work2))
    bound = np.concatenate((bound1, bound2))
    param_m, param_n = work.shape

    # configuration parameters
    args = configuration()
    args.init_mat = 'id_index'
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

