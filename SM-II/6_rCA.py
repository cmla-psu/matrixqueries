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
from softmax import gm0_variance
from convexdp import ConvexDP, ca_variance, wCA, wCA2


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


def test():
    """Test workload."""
    n = 64
    # work = np.array([[1, 1], [0, 1], [1, 0]])
    work = workload(n)
    # bound = np.array([20, 5, 1, 10, 1])
    bound = np.ones(n)*4
    bound[0:1] = 1
    # bound[-10:] = 10
    return work, bound


def test1():
    """Test workload."""
    n = 64
    # work = np.array([[1, 1], [0, 1], [1, 0]])
    work = workload(n)
    # bound = np.array([20, 5, 1, 10, 1])
    bound = np.ones(n)
    bound[0:5] = 2
    # bound[-10:] = 10
    return work, bound


def test_rca():
    """Test rCA methods."""
    n = 100*10
    k = 4
    work = np.zeros((n+1, n))
    work[:n, :] = np.eye(n)
    work[n, :] = 1
    bound = np.ones(n+1)
    bound[n] = k
    return work, bound


def WRelated(param_m, param_n, param_s):
    """Related workload."""
    mat_a = np.random.normal(0, 1, [param_s, param_n])
    mat_c = np.random.normal(0, 1, [param_m, param_s])
    work = mat_c @ mat_a
    # bound = np.ones(param_m)
    bound = np.random.randint(1, 10, param_m)
    # bound[1:5] = 1
    return work, bound


if __name__ == '__main__':
    start = time.time()
    np.random.seed(0)
    work, bound = test_rca()
    param_m, param_n = np.shape(work)

    # configuration parameters
    args = configuration()
    args.init_mat = 'id_index'
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
    # pcost = 1
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
