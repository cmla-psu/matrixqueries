# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 22:04:24 2020.

@author: 10259

Configuration file
"""

import argparse


def configuration():
    """
    Return configuration parameters.

    Returns
    -------
    args : parameters.

    """
    parser = argparse.ArgumentParser(description='Matrix Query')

    parser.add_argument('--maxiter', default=3000, help='total iteration')
    parser.add_argument('--maxitercg', default=3,
                        help='maximum iteration for conjugate gradient method')
    parser.add_argument('--maxiterls', default=3,
                        help='maximum iteration for finding a step size')
    parser.add_argument('--theta', default=1e-10,
                        help='determine when to stop conjugate gradient method'
                        ' smaller theta makes the step direction more accurate'
                        ' but it takes more time')
    parser.add_argument('--beta', default=0.5, help='step size decrement')
    parser.add_argument('--sigma', default=1e-2,
                        help='determine how much decrement in '
                        'objective function is sufficient')
    parser.add_argument('--NTTOL', default=1e-3,
                        help='determine when to update t, '
                        'smaller NTTOL makes the result more accurate '
                        'and the convergence slower')
    parser.add_argument('--TOL', default=1e-3,
                        help='determine when to stop the whole program')
    parser.add_argument('--MU', default=2, help='increment for '
                        'barrier approximation parameter t')
    return parser.parse_args()
