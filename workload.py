# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 22:49:58 2020.

@author: 10259

Generates hb tree structures and the B/V matrices for different problems
"""

import numpy as np


def matrix_workload(n: int) -> np.ndarray:
    """Create the workload matrix of example 3."""
    mat_w = np.zeros([n, n])
    for i in range(n):
        mat_w[i, :n-i] = 1
    # mat_w = np.tril(np.ones([n, n]), 0)
    return mat_w


def matrix_basis(n: int) -> np.ndarray:
    """Create the basis matrix."""
    return matrix_workload(n)


def matrix_index(n: int) -> np.ndarray:
    """
    Create the index matrix.

    mat_index = mat_workload @ mat_basis^{-1}
    """
    inv_mat_b = np.linalg.solve(matrix_basis(n), np.eye(n))
    mat_i = matrix_workload(n) @ inv_mat_b
    return mat_i


def variance_bound(n: int) -> np.ndarray:
    """Create the variance bound."""
    return np.ones(n)


def hb_strategy_matrix(n: int, k: int) -> np.ndarray:
    """
    Strategy matrix of HB method.

    here requires n is a square number so that n = k*k
    special case : if n=2, return a given matrix.

    Parameters
    ----------
    n :
        the number of nodes
    k :
        the branching factor
    """
    if n == 2:
        strategy = np.array([[1, 1], [1, 0], [0, 1]])
    else:
        m = 1 + k + n
        strategy = np.zeros([m, n])
        strategy[0, :] = 1
        for i in range(k):
            strategy[i+1, k*i: k*i+k] = 1
        strategy[k+1:, :] = np.eye(n)
    return strategy


def hb_variance(mat_w: np.ndarray, strategy: np.ndarray, p_cost: float) \
                -> float:
    """
    Calculate the maximum variance for a single query in workload.

    Use HB method with privacy cost at most p_cost.

    Parameters
    ----------
    strategy :
        the hb tree structure
    p_cost :
        the privacy cost

    Returns
    -------
    variance :
        maximum variance for a single query
    """
    m, n = np.shape(strategy)
    mat_identity = np.eye(m)
    # pseudoinverse of matrix strategy
    pseudo_inv_strategy = np.linalg.pinv(strategy)

    # how much variance for Gaussian Mechnism we need under the privacy cost
    # l2_sensitivity = np.linalg.norm(strategy, ord=2, axis=0)
    l2_sensitivity = 3
    sigma_square = l2_sensitivity/p_cost

    # calculate maximum variance anwsering a single query
    mat_var = mat_w @ pseudo_inv_strategy
    invcov = sigma_square*mat_identity
    variance = np.max(mat_var @ invcov @ mat_var.T)
    return variance
