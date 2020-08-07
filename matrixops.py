# -*- coding: utf-8 -*-
"""General purpose matrix utilities."""

import numpy as np


def is_pos_def(mat: np.ndarray) -> bool:
    """Return True if the numpy array A is symmetric positive definite."""
    pd = False
    # first check symmetry
    if np.allclose(mat, mat.T, rtol=1e-5, atol=1e-8):
        # check if Cholesky decomposition is successful
        try:
            np.linalg.cholesky(mat)
            pd = True
        except np.linalg.LinAlgError:
            pd = False
    return pd


def matrix_3d_broadcasting(mat: np.ndarray) -> np.ndarray:
    """
    Generate 3d matrix mat_3d[i,:,:]=mat[:,].T @ mat[:,].

    Parameters
    ----------
    mat : np.ndarray
        Input matrix.

    Returns
    -------
    np.ndarray
        3d matrix.

    """
    # 3d vector of size (m,n,1)
    mat_row = mat[:, :, None]
    # 3d vector of size (m,1,n)
    mat_col = mat[:, None, :]
    # 3d vector of size (m,n,n)
    # for each row vector v = mat[i,:], mat_3d[i,:,:] = v.T @ v
    mat_3d = np.einsum('mnr,mrd->mnd', mat_row, mat_col)
    return mat_3d
