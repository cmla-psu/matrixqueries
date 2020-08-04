# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 08:00:58 2020

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

def funf2(B,iX):
    '''
    Inequality constraint function for privacy cost
     
    Parameters
    ----------
    B : the basis matrix
    iX : the inverse of the co-variance matrix X
    '''
    BXB = B.T @ iX @ B
    d = np.diag(BXB)
    return d

def obj(X,s,t,c,V,B):
    '''
    Objective function
     
    Parameters
    ----------
    X : co-variance matrix
    s : controls the accuracy of the barrier approximation
    t : controls the approximation to max function
    c : variance bound, V: index matrix, B: basis matrix
    '''
    _,n = np.shape(V)
    iX = np.linalg.solve(X, np.eye(n))
    f1 = funf1(c,V,X)
    f2 = funf2(B,iX)
    F = s*np.log(np.sum(np.exp(t*f2))) - np.sum(np.log(f1))
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

def Derivatives(c,s,t,V,X,B,iX):
    '''
    Return Gradient and Hessian information
     
    Parameters
    ----------
    c : variance bound
    s : barrier approximation parameter
    t : softmax approximation parameter
    V : index matrix of size (m,n)
    B : basis matrix of size (n,n)
    X: co-variance matrix, iX: inverse of X
     
    Returns
    -------
    G : gradient vector [vec(dF/dX)]
    Vmat : 3d vector of size (m,n,n)
        Vmat[i,:,:] = V[i,:].T @ V[i,:]
        storage the gradient of VXV.T 
    XBX : 3d vector of size (n,n,n)
        XBX[i,:,:] = -iX @ Bmat[i,:,:] @ iX
        storage the gradient of B.T X^{-1} B
    f1,f2 : inequality functions
    '''
    
    # vectorization 
    # Gradient V*V.T
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
    
    # Gradient -iX*B.T*B*iX
    f2 = funf2(B,iX)
    # 3d vector of size (n,n,1)
    Bcol = (B.T)[:,:,None]
    # 3d vector of size (n,1,n)
    Brow = (B.T)[:,None,:]
    # 3d vector of size (n,n,n)
    # for each column vector b = B[:,i], Bmat[i,:,:] = b @ b.T
    Bmat = np.einsum('mnr,mrd->mnd',Bcol,Brow)
    # XBX[i,:,:] = -iX @ Bmat[i,:,:] @ iX
    XBX = -iX @ Bmat @ iX
    # gf2[i] = XBX[i,:,:]*exp(t*f2[i])
    # component of gradient df2/dX
    gf2 = XBX*np.exp(t*f2[:,None,None])
    
    # G is a n*n gradient matrix
    G = np.sum(gf1,0) + np.sum(gf2,0)*s*t/np.sum(np.exp(t*f2))
    # flatten G2 into vector 
    # G is an 1d vector of size n*n
    G = G.flatten('F')
    
    return G, Vmat, XBX, f1, f2


def Hx(f1,f2,s,t,Vmat,iX,XBX,x):
    '''
    Calculate the multiplication d = H*x.
    Use kron(A,B)*vec(C) = vec(BCA) to avoid using kronecker product,
    here matrix A is symmetric so A = A.T
    
    d = H*x
    
    H consists of three kronecker products, 
    we will use matrix product instead
    
    Parameters
    ----------
    f1 : inequality constraint function for variance
    f2 : inequality constraint function for privacy cost
    s : barrier approximation parameter
    t : softmax approximation parameter
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
    X = np.reshape(x,[n,n],'F')
    # calculate kron(Vmat,Vmat)*x
    hf1mat = Vmat @ X @ Vmat
    hf1 = hf1mat/(f1**2)[:,None,None]
    
    denominator = np.sum(np.exp(t*f2))
    gf2 = XBX*np.exp(t*f2[:,None,None])
    gf2mat = np.sum(gf2,0)
    # calculate kron(gf2mat,gf2mat)*x
    hf2 = gf2mat @ X @ gf2mat
    
    # calculate -kron(XBX,XBX)*x - kron(XBX,iX)*x - kron(iX,XBX)*x
    hf22mat = XBX @ X @ XBX - iX @ X @ XBX - XBX @ X @ iX
    hf22 = hf22mat*np.exp(t*f2[:,None,None])
    Hx =  np.sum(hf1,0) - hf2*s*t**2/denominator**2 + np.sum(hf22,0)*s*t**2/denominator

    d = Hx.flatten('F')
    
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
    n = 4
    B = workload(n)
    V = np.eye(n)
#    B = np.eye(n)
#    V = workload(n)
    c = np.ones(n)
    
    s = 1
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
    # MU : increament for s 
    MU = 50
    # PI : Increament for t
    PI = 5
    # Initial values
    I = np.eye(n);
    X = 0.8*I
    iX = np.linalg.solve(X,I)
    BXB = B.T @ iX @ B
    
    history = []
    
    for iters in range(maxiter):        
        [G,Vmat,XBX,f1,f2] = Derivatives(c,s,t,V,X,B,iX)
        
        v = np.zeros(n*n);
 
        r = -G
        p = r;
        rsold = r.dot(r);
        for i in range(maxitercg):
            Hp = Hx(f1,f2,s,t,Vmat,iX,XBX,p)
            a = rsold/(p @ Hp);
            v = v + a*p;
            r = r - a * Hp;
            rsnew = r.dot(r);
    
            if rsnew < theta:
                break
            b = rsnew/rsold;
            p = r + b*p;
            rsold = rsnew;
        
        # step size
        delta = v.dot(G); Xold = X;
        fcurr = obj(X,s,t,c,V,B)
        flast = fcurr; history.append(fcurr)
        
        # Stop the algorithm when criteria are met
        if (np.abs(delta) < NTTOL):
            gap = 5*n/s
            if (gap < TOL):
                break;
            s = MU*s
            t = PI*t
            print('update t: {0}'.format(s))
        else:
            for k in range(maxiterls):
                alpha = beta**(k-1); 
                step = np.reshape(v,[n,n],'F')
                X = Xold + alpha*step
                iX = np.linalg.solve(X,I)
                # check in domain
                f1 = funf1(c,V,X);
                f2 = funf2(B,iX);
                if(np.min(f1)<0):
                    continue
                # check positive definiteness
                if(not is_pos_def(X)):
                    continue
                fcurr=obj(X,s,t,c,V,B);
                if(fcurr<=flast+alpha*sigma*delta):
                    break

            print("iter:{0}, fobj:{1}, opt:{2}, cg:{3}, ls:{4}".
                  format(iters,fcurr,delta,i,k))   
    
#    print(X)
    pcost = np.max(f2)
    print('p =',pcost)
    A = hbn(n,np.int(np.sqrt(n)))
    var= hbvar(B,A,pcost)
    print('v =',var)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    