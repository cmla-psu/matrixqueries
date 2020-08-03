# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 09:25:16 2020

@author: 10259
"""

import numpy as np

def funf1(c,V,X):
    '''
    Inequality constraint function for variance
     
    Parameters
    ----------
    c : variance bound 
    V : the index matrix
    X : the co-variance matrix 
    '''
    VXV = V @ X @ V.T
    d = np.diag(VXV)
    return c - d

def funf2(s,B,iX):
    '''
    Inequality constraint function for privacy cost
     
    Parameters
    ----------
    s : the privacy cost
    B : the basis matrix
    iX : the inverse of the co-variance matrix X
    '''
    BXB = B.T @ iX @ B
    d = np.diag(BXB)
    n,_ = np.shape(B)
    return s*np.ones(n) - d

def obj(Y,t,c,V,B):
    '''
    Objective function
     
    Parameters
    ----------
    Y = [s;vec(X)]
    s : privacy cost 
    X : co-variance matrix
    t : controls the accuracy of the barrier approximation
    c : variance bound, V: index matrix, B: basis matrix
    '''
    _,n = np.shape(V)
    s = Y[0]
    X = np.reshape(Y[1:],[n,n],'F')
    iX = np.linalg.solve(X, np.eye(n))
    f1 = funf1(c,V,X)
    f2 = funf2(s,B,iX)
    F = s*t - np.sum(np.log(f1)) - np.sum(np.log(f2))
    return F

def is_pos_def(A):
    '''
    Check positive definiteness 
    '''
    # first check symmetry 
    if np.allclose(A,A.T,1e-5,1e-8):
        # whether cholesky decomposition is successful
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False

def workload(n):
    '''
    Create the workload matrix of example 3
    '''
    W = np.zeros([n,n])
    for i in range(n):
        W[i,:n-i] = 1
    return W

def Derivatives(c,t,s,V,X,B,iX):
    '''
    Return Gradient and Hessian information
     
    Parameters
    ----------
    c : variance bound
    t : barrier approximation parameter
    V : index matrix of size (m,n)
    B : basis matrix of size (n,n)
    s : privacy cost, X: co-variance matrix, iX: inverse of X
     
    Returns
    -------
    G : gradient vector [dF/ds; vec(dF/dX)]
    Vmat : 3d vector of size (m,n,n)
        Vmat[i,:,:] = V[i,:].T @ V[i,:]
        storage the gradient of VXV.T 
    XBX : 3d vector of size (n,n,n)
        XBX[i,:,:] = -iX @ Bmat[i,:,:] @ iX
        storage the gradient of B.T X^{-1} B
    f1,f2 : inequality functions
    '''
    
    f1 = funf1(c,V,X)
    # 3d vector of size (m,n,1)
    Vcol = V[:,:,None]
    # 3d vector of size (m,1,n)
    Vrow = V[:,None,:]
    # 3d vector of size (m,n,n)
    # for each row vector v = V[i,:], Vmat[i,:,:] = v.T @ v
    Vmat = np.einsum('mnr,mrd->mnd',Vcol,Vrow)
    # gf1[i] = Vmat[i,:,:]/f1[i]
    # component gradient df1/dX
    gf1 = Vmat/f1[:,None,None]
    
    f2 = funf2(s,B,iX)
    # 3d vector of size (n,n,1)
    Bcol = (B.T)[:,:,None]
    # 3d vector of size (n,1,n)
    Brow = (B.T)[:,None,:]
    # 3d vector of size (n,n,n)
    # for each column vector b = B[:,i], Bmat[i,:,:] = b @ b.T
    Bmat = np.einsum('mnr,mrd->mnd',Bcol,Brow)
    # XBX[i,:,:] = -iX @ Bmat[i,:,:] @ iX
    XBX = -iX @ Bmat @ iX
    # gf2[i] = XBX[i,:,:]/f2[i]
    # component of gradient df2/dX
    gf2 = XBX/f2[:,None,None]
    
    # gradient dF/ds
    G1 = t - np.sum(1/f2)
    # gradient dF/dX
    G2 = np.sum(gf1,0) + np.sum(gf2,0)
    # flatten G2 into vector 
    # G is an 1d vector of size 1+n*n
    G = np.concatenate([[G1],G2.flatten('F')])

    return G, Vmat, XBX, f1, f2


def Hx(f1,f2,Vmat,iX,XBX,x):
    '''
    Calculate the multiplication d = H*x.
    Use kron(A,B)*vec(C) = vec(BCA) to avoid using kronecker product,
    here matrix A is symmetric so A = A.T
    
    H = [H11,H21.T;H21,H22]
    H11 : float, H21 : column vector of size n*n
    H22 : matrix of size (n*n,n*n)
    
    x = [x1;x2], d = [d1;d2]
    x1,d1: float, x2,d2: vector of size n*n
    
    d1 = H11*x1 + dot(H21.T,d2)
    d2 = H21*x1 + H22*x2
    
    H22 consists of three kronecker products, 
    we will use matrix product instead
    
    Parameters
    ----------
    f1 : inequality constraint function for variance
    f2 : inequality constraint function for privacy cost
    Vmat : 3d vector of size (m,n,n)
        Vmat[i,:,:] = V[i,:].T @ V[i,:]
        storage the gradient of VXV.T 
    XBX : 3d vector of size (n,n,n)
        XBX[i,:,:] = -iX @ Bmat[i,:,:] @ iX
        storage the gradient of B.T X^{-1} B
    iX : inverse of co-variance matrix X
    x : input vector to multiply with Hessian matrix H
        
    Returns
    -------
    d : the return calculation of H*x    
    '''
    
    n,_ = np.shape(iX)
    x1 = x[0]
    x2 = x[1:]
    
    # H11 is the second derivitive w.r.t. s: d^2 F/d s^2
    H11 = np.sum(1/f2**2)
    # H21 is the partial derivitive d^2 F/ ds dX
    hf = -XBX/(f2**2)[:,None,None]
    H21 = np.sum(hf,0)
    H21 = H21.flatten('F')
    d1 = H11*x1 + H21.dot(x2)
    
    Y = np.reshape(x2,[n,n],'F')
    # calculate kron(Vmat,Vmat)*x2
    hf1mat = Vmat @ Y @ Vmat
    # calculate kron(XBX,XBX)*x2
    hf2mat = XBX @ Y @ XBX
    # calculate -kron(XBX,iX)*x2 - kron(iX,XBX)*x2
    hf22mat = -iX @ Y @ XBX - XBX @ Y @ iX
    
    # second derivatives w.r.t. X
    hf1 = hf1mat/(f1**2)[:,None,None]
    hf2 = hf2mat/(f2**2)[:,None,None]
    hf22 = hf22mat/f2[:,None,None]
    
    # H22x2 is the product H22*x2
    H22x2 =  np.sum(hf1,0) + np.sum(hf2,0) + np.sum(hf22,0)
    # transfer the matrix into vector form
    H22x2 = H22x2.flatten('F')
    d2 = H21*x1 + H22x2
    
    d = np.concatenate([[d1],d2])
    
    return d
    
def hbn(n,k):
    '''
    Strategy matrix of HB method
    here requires n is a square number so that n = k*k
    special case : if n=2, return a given matrix 
    
    Parameters
    ----------
    n : the number of nodes
    k : the branching factor
    '''
    if n==2:
        Tree = np.array([[1,1],[1,0],[0,1]])
    else:
        m = 1 + k + n
        Tree = np.zeros([m,n])
        Tree[0,:] = 1
        for i in range(k):
            Tree[i+1,k*i:k*i+k] = 1
        Tree[k+1:,:] = np.eye(n)
    return Tree

def hbvar(W,A,s):
    '''
    Calculate the maximum variance for a single query in workload W
    using HB method with privacy cost at most s.
    
    Parameters 
    ----------
    A is the hb tree structure
    s is the privacy cost
    '''
    m,n = np.shape(A)
    I = np.eye(m)
    # pseudoinverse of matrix A
    pA = np.linalg.pinv(A)
    sigma = np.sqrt(3/s)
    Var = W @ pA
    BXB = Var @ (sigma**2*I) @ Var.T
    a = np.max(np.diag(BXB))
    return a
    

if __name__ == '__main__':
    n = 64
    B = workload(n)
    V = np.eye(n)
#    B = np.eye(n)
#    V = workload(n)
    c = np.ones(n)
    
    t = 1
    # maxiter is the total iteration 
    maxiter = 3000;
    # maxitercg is the maximum iteration for conjugate gradient method
    maxitercg = 5
    # maiterls is the maximum iteration for finding a step size
    maxiterls = 50
    
    # theta : determine when to stop conjugate gradient method
    # smaller theta makes the step direction more accurate 
    # but it takes more time
    theta = 1e-3
    # beta : step size decreament
    beta = 0.5
    # sigma : determine how much decreament in objective function is sufficient
    sigma = 1e-4
    # NTTOL : determine when to update t
    # smaller NTTOL makes the result more accurate and the convergence slower
    NTTOL = 1e-2
    # NTTOL : determine when to stop the whole programe 
    TOL = 1e-2
    # MU : increament for t 
    MU = 50
    
    # Initial values
    I = np.eye(n)
    # a good initialization of X will make convergence faster
    X = 0.8*I
    iX = np.linalg.solve(X,I)
    BXB = B.T @ iX @ B
    s = 1.01*np.max(np.diag(BXB))
    
    history = []
    Y = np.concatenate([[s],X.flatten('F')])
    
    for iters in range(maxiter):
        
        [G,Vmat,XBX,f1,f2] = Derivatives(c,t,s,V,X,B,iX)
        
        # conjugate gradient method 
        # find the newton's direction v
        v = np.zeros(n*n + 1);
        
        r = -G
        p = r
        rsold = r.dot(r);
        for i in range(maxitercg):
            Hp = Hx(f1,f2,Vmat,iX,XBX,p)
            a = rsold/(p @ Hp);
            v = v + a*p;
            r = r - a * Hp;
            rsnew = r.dot(r);
    
            if rsnew < theta:
                break
            b = rsnew/rsold;
            p = r + b*p;
            rsold = rsnew;
        
        delta = v.dot(G); Yold = Y;
        fcurr = obj(Y,t,c,V,B)
        flast = fcurr; history.append(fcurr)
        
        # Stop the algorithm when criteria are met
        if (np.abs(delta) < NTTOL):
            gap = 5*n/t
            if (gap < TOL):
                break;
            t = MU*t
            print('update t: {0}'.format(t))
        else:
            # find a proper step size
            for k in range(maxiterls):
                alpha = beta**(k-1); 
                Y = Yold + alpha*v
                s = Y[0]
                X = np.reshape(Y[1:],[n,n],'F')
                iX = np.linalg.solve(X,I)
                # check feasibility
                f1 = funf1(c,V,X);
                f2 = funf2(s,B,iX);
                if(np.min([f1,f2])<0):
                    continue
                # check positive definiteness
                if(not is_pos_def(X)):
                    continue
                fcurr=obj(Y,t,c,V,B)
                # if there is sufficient decreasement then stop
                if(fcurr<=flast+alpha*sigma*delta):
                    break

            print("iter:{0}, fobj:{1}, opt:{2}, cg:{3}, ls:{4}".
                  format(iters,fcurr,delta,i,k))   
    
    print('s =',s)
    A = hbn(n,np.int(np.sqrt(n)))
    var= hbvar(B,A,s)
    print('v =',var)