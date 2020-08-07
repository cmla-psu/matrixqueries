# -*- coding: utf-8 -*-
"""Contains code for privacy analysis."""

import numpy as np


def basic_l2_privacy_cost_vector(strategy: np.ndarray, noise_scale: float)\
                                -> np.ndarray:
    """ Returns the column-wise privacy cost vector of the strategy matrix
    when iid noise is added, this is the same as the vector of l2 column norms

    Input:
       strategy: the query matrix to which noise would be added in a dp mechanism
       noise_scale: the standard deviation of the independent Gaussian noise

    Returns:
        a row vector containing the privacy cost of each column. There are as
        many columns as in strategy
    """
    hit = np.linalg.norm(strategy, ord=2, axis=0) / noise_scale
    return hit


def basic_l2_privacy_cost(strategy: np.ndarray, noise_scale: float) -> float:
    """ Returns the privacy cost of the strategy matrix
    when iid noise is added, this is the same as the largest of the l2 column norms

    Input:
       strategy: the query matrix to which noise would be added in a dp mechanism
       noise_scale: the standard deviation of the independent Gaussian noise

    Returns:
        the privacy cost
    """
    cost = basic_l2_privacy_cost_vector(strategy, noise_scale)
    return cost.max()


def l2_privacy_cost_vector(strategy: np.ndarray, invcov: np.ndarray = None,
                           cov: np.ndarray = None) -> np.ndarray:
    """Return the privacy cost vector of the strategy matrix \
        when multivariate Gaussian noise is added.

    Inputs
    ------
       strategy:
           the query matrix to which noise \
           would be added in a dp mechanism
       invcov:
           the inverse of the covariance matrix
       cov:
           the positive definite covariance matrix

    Note: only cov or incov is needed (do not specify both)

    Returns
    -------
        a row vector containing the privacy cost of each column. There are as
        many columns as in strategy
    """
    # Do not specify both a covariance matrix and its inverse
    assert cov is None or invcov is None
    if invcov is None:
        invcov = np.linalg.solve(cov, np.eye(cov.shape[0]))
    # cost = np.sqrt(np.diag(strategy.T @ invcov @ strategy))
    cost = np.diag(strategy.T @ invcov @ strategy)
    return cost


def l2_privacy_cost(strategy: np.ndarray, invcov: np.ndarray = None,
                    cov: np.ndarray = None) -> float:
    """Return the privacy cost of the strategy matrix.

    When iid noise is added, this is the same as the largest
    of the l2 column norms

    Inputs
    ------
       strategy:
           the query matrix to which noise would be added \
           in a dp mechanism
       noise_scale:
           the standard deviation of the independent Gaussian noise

    Returns
    -------
        the privacy cost
    """
    cost = l2_privacy_cost_vector(strategy, invcov=invcov, cov=cov)
    return cost.max()
