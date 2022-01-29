# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 05:04:30 2020.

@author: 10259

Newton Approximation.
"""

import numpy as np
# import cupy as np
# from softmax import WRange, WRelated


def is_pos_def(A):
    """
    Check positive definiteness.

    Return true if psd.
    """
    zero = np.zeros_like(A)
    # first check symmetry
    if np.allclose(A, A.T, 1e-5, 1e-8):
        # whether cholesky decomposition is successful
        try:
            L = np.linalg.cholesky(A)
            return L, True
        except np.linalg.LinAlgError:
            return zero, False
    else:
        return zero, False


def Hx(iX, S, G):
    """Hess tiems x."""
    mat = -iX @ S @ G - G @ S @ iX
    return mat


def ConvexDP(W):
    """Newton Approximation."""
    _, n = W.shape
    maxiter = 30
    maxiterls = 50
    maxitercg = 5
    theta = 1e-3
    accuracy = 1e-5
    beta = 0.5
    sigma = 1e-4

    X = np.eye(n)
    iD = np.eye(n)
    V = W.T @ W
    V = V + theta*np.mean(np.diag(V)) * iD
    A = np.linalg.cholesky(X)
    iX = np.linalg.solve(A.T, np.linalg.solve(A, iD))
    G = -iX @ V @ iX
    fcurr = np.sum(V * iX)
    history = []

    for iteration in range(maxiter):
        # find search direction
        if iteration == 0:
            D = -G
            np.fill_diagonal(D, 0)
            i = -1
        else:
            D = np.zeros([n, n])
            R = -G - Hx(iX, D, G)
            np.fill_diagonal(R, 0)
            P = R
            rsold = np.sum(R * R)
            for i in range(maxitercg):
                Hp = Hx(iX, P, G)
                alpha = rsold/np.sum(P * Hp)
                D = D + alpha*P
                np.fill_diagonal(D, 0)
                R = R - alpha*Hp
                np.fill_diagonal(R, 0)
                rsnew = np.sum(R * R)
                if rsnew < 1e-10:
                    break
                P = R + rsnew/rsold * P
                rsold = rsnew

        # find step size
        delta = np.sum(D * G)
        Xold = X
        flast = fcurr
        history.append(fcurr)
        for j in range(maxiterls):
            alpha = beta**j
            X = Xold + alpha*D
            A, flag = is_pos_def(X)
            if flag:
                iX = np.linalg.solve(A.T, np.linalg.solve(A, iD))
                G = -iX @ V @ iX
                fcurr = np.sum(V * iX)
            if fcurr <= flast + alpha * sigma * delta:
                break
        # print(iteration, fcurr, np.linalg.norm(D), i, j)

        if i == maxiterls:
            X = Xold
            fcurr = flast
            break
        if np.abs((flast-fcurr)/flast) <= accuracy:
            break

    return np.linalg.cholesky(X)


def ca_variance(W, A, s):
    """
    Calculate the maximum variance for a single query in workload W.

    using HB method with privacy cost at most s.

    Parameters
    ----------
    A is the hb tree structure
    s is the privacy cost
    """
    m, n = A.shape
    mI = np.eye(m)
    # inverse of matrix A
    pA = np.linalg.inv(A)
    sigma = np.sqrt(1/s)
    Var = W @ pA.T
    BXB = Var @ (sigma**2*mI) @ Var.T
    a = np.diag(BXB)
    return a


def WRange(param_m, param_n):
    """Range Workload."""
    work = np.zeros([param_m, param_n])
    for i in range(param_m):
        num_1 = np.random.randint(param_n)
        num_2 = np.random.randint(param_n)
        if num_1 < num_2:
            low = num_1
            high = num_2
        else:
            low = num_2
            high = num_1
        work[i, low:high+1] = 1
    return work


def seg_max(vec, k):
    """Find the maximum for each k elements."""
    mat = np.reshape(vec, [-1, k])
    seg = np.max(mat, 1)
    return seg


def WDiscrete(param_m, param_n, prob):
    """Discrete workload."""
    # work = np.zeros([param_m, param_n])
    p = [1-prob, prob]
    work = np.random.choice(2, [param_m, param_n], p)
    work = 2*work - 1
    return work


def WRelated(param_m, param_n, param_s):
    """Related workload."""
    mat_a = np.random.normal(0, 1, [param_s, param_n])
    mat_c = np.random.normal(0, 1, [param_m, param_s])
    work = mat_c @ mat_a
    return work


def wCA(work, bound, pcost):
    """Re-weighted workload for CA, w[i] = c[i]."""
    m, n = work.shape
    work_s = np.copy(work)
    work_s = work_s + 0.0
    for i in range(m):
        work_s[i, :] = work[i, :]/bound[i]
    strat = ConvexDP(work_s)
    var_s = ca_variance(work_s, strat, pcost)
    var_s = var_s * bound * bound
    # print("wvar=", np.max(var_s/bound))
    return strat, var_s


def wCA2(work, bound, pcost):
    """Re-weighted workload for CA, w[i] = sqrt(c[i])."""
    m, n = work.shape
    work_s = np.copy(work)
    work_s = work_s + 0.0
    for i in range(m):
        work_s[i, :] = work[i, :]/np.sqrt(bound[i])
    strat = ConvexDP(work_s)
    var_s = ca_variance(work_s, strat, pcost)
    var_s = var_s * bound
    # print("wvar=", np.max(var_s/bound))
    return strat, var_s


if __name__ == '__main__':
    np.random.seed(0)
    param_n = 128
    param_m = 2*param_n
    param_s = param_n // 2
    # W = WRange(param_m, param_n)
    W = WDiscrete(param_m, param_n, 0.2)
    # W = WRelated(param_m, param_n, param_s)
    iD = np.eye(param_n)
    V = W.T @ W
    A = ConvexDP(W)

    iX = np.linalg.solve(A.T, np.linalg.solve(A, iD))
    fcurr = np.sum(V * iX)
    # print(fcurr/param_m)
    name = 'ca_' + str(param_n) + '.npy'
    # np.save(name, A)

    # p_cost = {2: 1.333, 4: 1.9, 8: 2.483, 16: 3.237,
    #           64: 5.642, 256: 8.862, 1024: 13.008}
    var = ca_variance(W, A, 44.59)
    print("var=", np.max(var))
