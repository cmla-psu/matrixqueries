# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 12:25:51 2020.

@author: 10259

Commonly used functions.
"""

import numpy as np
# import privacy
import matrixops


def func_var(var_bound: np.ndarray, mat_index: np.ndarray,
             cov: np.ndarray) -> np.ndarray:
    """
    Inequality constraint function for variance.

    Parameters
    ----------
    var_bound : np.ndarray
        Variance bound.
    mat_index : np.ndarray
        The index matrix.
    cov : np.ndarray
        The co-variance matrix.

    Returns
    -------
    Inequality constraint function for variance.

    """
    vec_var = ((mat_index @ cov) * mat_index).sum(axis=1)
    return var_bound - vec_var


def func_pcost(p_cost: float, mat_basis: np.ndarray, invcov: np.ndarray = None,
               cov: np.ndarray = None) -> np.ndarray:
    """
    Inequality constraint function for privacy cost.

    Parameters
    ----------
    p_cost : float
        Privacy cost.
    mat_basis : np.ndarray
        Basis matrix.
    invcov : np.ndarray, optional
        Inverse of covariance matrix. The default is None.
    cov : np.ndarray, optional
        covariance matrix. The default is None.

    Returns
    -------
    Inequality constraint function for privacy cost.

    """
    # Do not specify both a covariance matrix and its inverse
    assert cov is None or invcov is None
    if invcov is None:
        invcov = np.linalg.solve(cov, np.eye(cov.shape[0]))
    # vec_cost = privacy.l2_privacy_cost_vector(mat_basis, invcov)
    # vec_cost_square = np.square(vec_cost)
    vec_cost = ((mat_basis.T @ invcov) * mat_basis.T).sum(axis=1)
    return p_cost - vec_cost


def func_obj(variable: np.ndarray, param_t: float, var_bound: np.ndarray,
             mat_index: np.ndarray, mat_basis: np.ndarray) -> np.ndarray:
    """
    Objective function.

    Parameters
    ----------
    variable : np.ndarray
        Combination of [pcost;vec(cov)].
    param_t : float
        Approximation parameter t.
    var_bound : np.ndarray
        Variance bound.
    mat_index : np.ndarray
        Index matrix.
    mat_basis : np.ndarray
        Basis matrix.

    Returns
    -------
    f_obj : np.ndarray
        Objetive function.

    """
    _, size_n = np.shape(mat_index)
    pcost = variable[0]
    cov = np.reshape(variable[1:], [size_n, size_n], 'F')
    invcov = np.linalg.solve(cov, np.eye(size_n))
    f_var = func_var(var_bound, mat_index, cov)
    f_pcost = func_pcost(pcost, mat_basis, invcov)
    f_obj = pcost*param_t - np.sum(np.log(f_var)) - np.sum(np.log(f_pcost))
    return f_obj


def gradient(variable: np.ndarray, param_t: float, var_bound: np.ndarray,
             mat_index: np.ndarray, mat_basis: np.ndarray,
             invcov: np.ndarray = None):
    """
    Calculate gradient and hessian information.

    Parameters
    ----------
    variable : np.ndarray
        Combination of [pcost;vec(cov)].
    param_t : float
        Approximation parameter t.
    var_bound : np.ndarray
        Variance bound.
    mat_index : np.ndarray
        Index matrix.
    mat_basis : np.ndarray
        Basis matrix.

    Returns
    -------
    g_total : np.ndarray
        Gradient vector [dF/ds; vec(dF/dX)].
    mat_v_3d : np.ndarray
        3d vector of size (m,n,n)
        storage the gradient of VXV.T
    mat_xbx_3d : np.ndarray
        3d vector of size (n,n,n)
        storage the gradient of B.T X^{-1} B

    """
    _, size_n = np.shape(mat_index)
    pcost = variable[0]
    cov = np.reshape(variable[1:], [size_n, size_n], 'F')
    if(invcov is None):
        invcov = np.linalg.solve(cov, np.eye(size_n))

    f_var = func_var(var_bound, mat_index, cov)
    mat_v_3d = matrixops.matrix_3d_broadcasting(mat_index)
    # g_f_var[i] = mat_v_3d[i,:,:]/f_var[i]
    # component gradient df1/dX
    g_f_var = mat_v_3d/f_var[:, None, None]

    f_pcost = func_pcost(pcost, mat_basis, invcov)
    mat_b_3d = matrixops.matrix_3d_broadcasting(mat_basis.T)
    # mat_xbx_3d[i,:,:] = -invcov @ mat_b_3d[i,:,:] @ invcov
    mat_xbx_3d = 0 - invcov @ mat_b_3d @ invcov
    # g_f_pcost[i] = mat_xbx_3d[i,:,:]/f_pcost[i]
    # component of gradient df2/dX
    g_f_pcost = mat_xbx_3d/f_pcost[:, None, None]

    # gradient dF/ds
    g_s = param_t - np.sum(1/f_pcost)
    # gradient dF/dX
    g_cov = np.sum(g_f_var, 0) + np.sum(g_f_pcost, 0)
    # flatten g_part_2 into vector
    # g_total is an 1d vector of size 1+n*n
    g_total = np.concatenate([[g_s], g_cov.flatten('F')])

    return g_total, mat_v_3d, mat_xbx_3d, f_var, f_pcost


def hessian_times_x(invcov: np.ndarray, mat_v: np.ndarray, mat_xbx: np.ndarray,
                    f_var: np.ndarray, f_pcost: np.ndarray,
                    vec: np.ndarray) -> np.ndarray:
    """
    Calculate the multiplication direction = H*vec.

    Use kron(A,B)*vec(C) = vec(BCA) to avoid using kronecker product,
    here matrix A is symmetric so A = A.T

    H = [H11,H21.T;H21,H22]
    H11 : float, H21 : column vector of size n*n
    H22 : matrix of size (n*n,n*n)

    vec = [vec_s;vec_cov], direction = [d_s;d_cov]
    vec_s,d_s: float, vec_cov,d_cov: vector of size n*n

    d_s = H11*vec_s + dot(H21.T,d_cov)
    d_cov = H21*vec_s + H22*vec_cov

    H22 consists of three kronecker products,
    we will use matrix product instead

    Parameters
    ----------
    invcov : np.ndarray
        Inverse of covariance matrix.
    mat_v : np.ndarray
        Storage the gradient of VXV.T.
    mat_xbx : np.ndarray
        Storage the gradient of B.T X^{-1} B
    f_var : np.ndarray
       Variance function.
    f_pcost : np.ndarray
        Privacy cost function.
    vec : np.ndarray
        Input vecotr.

    Returns
    -------
    direction : np.ndarray
        Output direction.

    """
    size_n, _ = np.shape(invcov)
    vec_s = vec[0]
    vec_cov = vec[1:]

    # mat_h_11 is the second derivitive w.r.t. s: d^2 F/d s^2
    mat_h_11 = np.sum(1/f_pcost**2)
    # mat_h_21 is the partial derivitive d^2 F/ ds dX
    h_f = -mat_xbx/(f_pcost**2)[:, None, None]
    mat_h_21 = np.sum(h_f, 0)
    mat_h_21 = mat_h_21.flatten('F')
    d_s = mat_h_11*vec_s + mat_h_21.dot(vec_cov)

    mat_cov = np.reshape(vec_cov, [size_n, size_n], 'F')
    # calculate kron(mat_v,mat_v)*vec_cov
    mat_kron_v = mat_v @ mat_cov @ mat_v
    # calculate kron(mat_xbx,mat_xbx)*vec_cov
    mat_kron_xbx = mat_xbx @ mat_cov @ mat_xbx
    # calculate -kron(mat_xbx,invcov)*vec_cov - kron(invcov,mat_xbx)*vec_cov
    mat_hess_xbx = -invcov @ mat_cov @ mat_xbx - mat_xbx @ mat_cov @ invcov

    # second derivatives w.r.t. X
    hess_v = mat_kron_v/(f_var**2)[:, None, None]
    hess_xbx = mat_kron_xbx/(f_pcost**2)[:, None, None]
    hessian = mat_hess_xbx/f_pcost[:, None, None]

    # hess_times_vec is the product H22*vec_cov
    hess_times_vec = np.sum(hess_v, 0) + np.sum(hess_xbx, 0) \
        + np.sum(hessian, 0)
    # transfer the matrix into vector form
    hess_times_vec = hess_times_vec.flatten('F')
    d_cov = mat_h_21*vec_s + hess_times_vec

    direction = np.concatenate([[d_s], d_cov])

    return direction
