import numpy as np


def is_pos_def(A: np.ndarray):
    """
    Returns True if the numpy array A is symmetric positive definite
    """
    pd = False
    # first check symmetry
    if np.allclose(A, A.T, rtol=1e-5, atol=1e-8):
        # check if Cholesky decomposition is successful
        try:
            np.linalg.cholesky(A)
            pd = True
        except np.linalg.LinAlgError:
            pd = False
    return pd
