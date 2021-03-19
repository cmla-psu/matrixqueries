# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 15:45:38 2021.

@author: 10259
"""

import numpy as np
from scipy.stats import norm


def func_phi(epsilon):
    """Calculate function phi."""
    num_1 = p_cost/2.0 - epsilon/p_cost
    num_2 = -p_cost/2.0 - epsilon/p_cost
    return norm.cdf(num_1)-np.exp(epsilon)*norm.cdf(num_2)


def find_epsilon(p_cost, delta=1e-9):
    """Find epsilon given delta."""
    epsilon = 1
    cdf = func_phi(epsilon)
    while cdf > delta:
        cdf = func_phi(epsilon)
        epsilon = epsilon + 0.1
    return epsilon


if __name__ == '__main__':
    p_cost_square = 8.77
    p_cost = np.sqrt(p_cost_square)
    epsilon = find_epsilon(p_cost)
    print(epsilon)
