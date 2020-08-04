"""
This is pytest testing code. Run pytest or pytest test_matrixops to run the test
"""

import matrixops
import numpy as np


def test_is_pos_def_symmetry():
    """ checks if matrixops.is_pos_def rejects non-symmetric matrices """
    A = np.array([[9, 8], [8.01, 9]])
    assert not matrixops.is_pos_def(A)


def test_is_pos_def_reject_psd():
    """ checks that matrixops.is_pos_def rejects semi-definite matrices """
    A = np.array([[3, 3, 3], [3, 3, 3], [3, 3, 3]])
    assert not matrixops.is_pos_def(A)


def test_is_pos_def_reject_nd():
    """ checks that matrixops.is_pos_def rejects negative definite matrices """
    A = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    assert not matrixops.is_pos_def(A)


def test_is_pos_def_reject_ind():
    """ checks that matrixops.is_pos_def rejects indefinite matrices """
    A = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    assert not matrixops.is_pos_def(A)


def test_is_pos_def_accept_pd():
    """ checks that matrixops.is_pos_def accepts positivedefinite matrices """
    n = 3  # number of rows in square matrix
    for _ in range(5):
        Q = np.random.normal(size=(n*n, n))
        A = Q.transpose() @ Q + 0.01 * np.eye(n)
        assert matrixops.is_pos_def(A)
