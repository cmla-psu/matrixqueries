# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 17:40:56 2020.

@author: 10259

Matrix Optimization Class
"""

import config
import workload
import common
import privacy
import matrixops
import numpy as np


class matrix_query:
    """Class for matrix query optimization."""

    def __init__(self, args, mat_basis=None, mat_index=None, var_bound=None):
        """
        Get Inputs.

        Parameters
        ----------
        args : TYPE
            Configuration parameters.
        mat_basis : np.ndarray
            Basis matrix. The default is None.
        mat_index : np.ndarray
            Index matrix. The default is None.
        var_bound : np.ndarray
            Variance bound. The default is None.

        Returns
        -------
        None.

        """
        self.size_n = np.shape(mat_basis)[0]
        self.mat_basis = mat_basis
        self.mat_index = mat_index
        self.var_bound = var_bound
        self.args = args
        self.mat_v = None
        self.mat_xbx = None
        self.gradient = None
        self.initialization()

    def initialization(self):
        """
        Initialize covariance matrix and privacy cost.

        Returns
        -------
        None.

        """
        self.mat_id = np.eye(self.size_n)
        self.param_t = 1
        self.cov = self.mat_id*0.7
        self.invcov = np.linalg.solve(self.cov, self.mat_id)
        self.pcost = 1.01*privacy.l2_privacy_cost(self.mat_basis, self.invcov)
        self.variable = np.concatenate([[self.pcost], self.cov.flatten('F')])

    def newton_direction(self):
        """
        Calculate newton's direction.

        Here we use conjugate gradient method to find the direction.

        Returns
        -------
        vec_v : np.ndarray
            Newtons' direction.
        i : int
            Output parameter.

        """
        # conjugate gradient method
        # find the newton's direction vec_v
        vec_v = np.zeros(self.size_n*self.size_n + 1)

        r = 0 - self.gradient
        p = r
        rsold = r.dot(r)
        for i in range(self.args.maxitercg):
            Hp = common.hessian_times_x(
                self.invcov, self.mat_v, self.mat_xbx,
                self.f_var, self.f_pcost, p)
            a = rsold/(p @ Hp)
            vec_v = vec_v + a*p
            r = r - a * Hp
            rsnew = r.dot(r)

            if rsnew < self.args.theta:
                break
            b = rsnew/rsold
            p = r + b*p
            rsold = rsnew
        return vec_v, i

    def step_size(self):
        """
        Find proper step size.

        Returns
        -------
        k : int
            Output parameter.
        fcurr : float
            Current objective function value.

        """
        # find a proper step size
        var_old = self.variable
        fcurr = common.func_obj(self.variable, self.param_t, self.var_bound,
                                self.mat_index, self.mat_basis)
        flast = fcurr
        for k in range(self.args.maxiterls):
            alpha = self.args.beta**(k)
            self.variable = var_old + alpha*self.step
            self.pcost = self.variable[0]
            self.cov = np.reshape(self.variable[1:],
                                  [self.size_n, self.size_n], 'F')
            self.invcov = np.linalg.solve(self.cov, self.mat_id)
            # check feasibility
            self.f_var = common.func_var(self.var_bound,
                                         self.mat_index, self.cov)
            self.f_pcost = common.func_pcost(self.pcost,
                                             self.mat_basis, self.invcov)
            if np.min([self.f_var, self.f_pcost]) < 0:
                continue
            # check positive definiteness
            if not matrixops.is_pos_def(self.cov):
                continue
            fcurr = common.func_obj(self.variable, self.param_t,
                                    self.var_bound, self.mat_index,
                                    self.mat_basis)
            # if there is sufficient decreasement then stop
            if fcurr <= flast + alpha * self.args.sigma * self.delta:
                break
        return k, fcurr

    def optimize(self):
        """
        Find a optimziae covariance matrix and compare with HB method.

        Returns
        -------
        None.

        """
        for iters in range(self.args.maxiter):
            [self.gradient, self.mat_v,
             self.mat_xbx, self.f_var,
             self.f_pcost] = common.gradient(self.variable, self.param_t,
                                             self.var_bound, self.mat_index,
                                             self.mat_basis, self.invcov)
            self.step, i = self.newton_direction()

            self.delta = self.step.dot(self.gradient)

            # Stop the algorithm when criteria are met
            if np.abs(self.delta) < self.args.NTTOL:
                gap = 5*self.size_n/self.param_t
                if gap < self.args.TOL:
                    break
                self.param_t = self.args.MU*self.param_t
                print('updatet: {0}'.format(self.param_t))
            else:
                k, fcurr = self.step_size()
                print("iter:{0}, fobj:{1}, opt:{2}, cg:{3}, ls:{4}".
                      format(iters, fcurr, self.delta, i, k))

        print('pcost =', self.pcost)
        hb_strategy = workload.hb_strategy_matrix(
            self.size_n, np.int(np.sqrt(self.size_n)))
        var = workload.hb_variance(self.mat_basis, hb_strategy, self.pcost)
        print('max_v =', var)


if __name__ == '__main__':
    param_n = 2
    basis = workload.matrix_basis(param_n)
    index = workload.matrix_index(param_n)
    bound = workload.variance_bound(param_n)

    # configuration parameters
    args = config.configuration()
    mat_opt = matrix_query(args, basis, index, bound)
    mat_opt.optimize()
