# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 14:54:23 2020.

@author: 10259
"""
import numpy as np


def workload(n, dtype=np.float32):
    """
    Create the workload matrix of example 3.

    Return the workload matrix.
    """
    # mat_w = np.zeros([n, n], dtype=dtype)
    # for i in range(n):
    #     mat_w[i, :n-i] = 1
    mat_w = np.triu(np.ones([n, n]))
    return mat_w


def age():
    """Dataset AGE."""
    first_1 = np.eye(2)
    second_1 = workload(116)
    work_gender = np.kron(first_1, second_1)

    first_2 = np.ones((1, 2))
    second_2 = workload(116)
    work_age = np.kron(first_2, second_2)

    add_1 = np.zeros((1, 232))
    add_1[18:116] = 1
    add_2 = np.zeros((1, 232))
    add_2[134:232] = 1
    add_3 = add_1 + add_2

    work_all = np.concatenate((work_gender, work_age, add_1, add_2, add_3))

    var = np.ones(116*3+3)*10

    var[98] = 1
    var[214] = 1
    var[330] = 1
    var[-3:] = 1

    return work_all, var


def hb_2_level(n):
    """
    Strategy matrix of HB method.

    here requires n is a square number so that n = k*k
    special case : if n=2, return a given matrix

    Parameters
    ----------
    n : the number of nodes
    k : the branching factor
    """
    if n == 2:
        Tree = np.array([[1, 1], [1, 0], [0, 1]])
    else:
        m = 1 + n
        Tree = np.zeros([m, n])
        Tree[0, :] = 1
        Tree[1:, :] = np.eye(n)
    # Tree = np.delete(Tree, 0, 0)
    return Tree


def hb_3_level(n, k):
    """
    Strategy matrix of HB method.

    here requires n is a square number so that n = k*k
    special case : if n=2, return a given matrix

    Parameters
    ----------
    n : the number of nodes
    k : the branching factor
    """
    if n == 2:
        Tree = np.array([[1, 1], [1, 0], [0, 1]])
    else:
        m = 1 + k + n
        Tree = np.zeros([m, n])
        Tree[0, :] = 1
        for i in range(k):
            Tree[i+1, k*i: k*i+k] = 1
        Tree[k+1:, :] = np.eye(n)
    # Tree = np.delete(Tree, 0, 0)
    return Tree


def hb_variance_2(W, A, s):
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
    # pseudoinverse of matrix A
    pA = np.linalg.pinv(A)
    sigma = np.sqrt(2/s)
    Var = W @ pA
    BXB = Var @ (sigma**2*mI) @ Var.T
    a = np.diag(BXB)
    return a


def hb_variance(W, A, s):
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
    # pseudoinverse of matrix A
    pA = np.linalg.pinv(A)
    sigma = np.sqrt(3/s)
    Var = W @ pA
    BXB = Var @ (sigma**2*mI) @ Var.T
    a = np.diag(BXB)
    return a


if __name__ == '__main__':
    np.random.seed(0)
    pcost = 1.63476145
    param_n = 232
    work, bound = age()

    k = 16
    hb_strategy = hb_3_level(param_n, k)
    hb_var = hb_variance(work, hb_strategy, pcost)
    print(np.max(hb_var/bound), end=" ")
