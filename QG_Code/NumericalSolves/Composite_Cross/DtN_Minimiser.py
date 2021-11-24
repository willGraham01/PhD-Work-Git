#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 14:21:17 2021

@author: will

This file allows the content of DtNEvalsViaMinimiser.ipynb to be called as a script from the command line.
For documentation, see the aforementioned file.
"""

# sys.argv[1:] contains command line arguments. Index 0 is just the script name
import argparse
# for returning values to the command line
import sys

import numpy as np
from numpy import exp, pi

from scipy.optimize import minimize, NonlinearConstraint

import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

import time
from datetime import datetime

#%% Runtime functions: converters, saves, loads, plots, etc

# Saves the output of a given solve to a file that can then be read in at a later time
def SaveRun(theta, omega, phiList, fname='', fileDump='./'):
    '''
    Saves the output of the minimisation proceedure (the eigenpairs) to a .npz file,
    which can then be read in by another script or notebook.
    INPUTS:
        theta: (2,) float, value of the quasi-momentum.
        omega: float, value of omega.
        phiList: list, elements are FourierFunction objects - this is the output of the minimisation proceedure.
        fname: str, the name of the file to which values will be saved.
        fileDump: str, path to the directory in which to save the .npz file.
    OUTPUTS:
        outFile: str, name of file to which values were saved
    '''

    M = ( phiList[-1].cMat.shape[0] - 1 ) // 2
    N = len(phiList)
    cVecStore = np.zeros((N, phiList[-1].cVec.shape[0]), dtype=complex)
    for n in range(N):
        cVecStore[n,:] = phiList[n].cVec

    if not fname:
        #automatically create file name
        timeStamp = datetime.now()
        outFile = fileDump + timeStamp.strftime("%Y-%m-%d_%H-%M_PhiList-M") \
			+ str(int(M)) + '-N' + str(N) + '.npz'
    else:
        outFile = fileDump + fname

    np.savez(outFile, theta=theta, omega=omega, M=int(M), cVecs=cVecStore, N=N)
    return outFile

# Conversion from cc (floats) to c (complex)
def Real2Comp(x):
    '''
    Given a vector x of length (2n,), which stores the real and imaginary parts of complex numbers z as
    z[j] = x[2j] + i*x[2j+1], return the vector z.
    INPUTS:
        x: (2n,) float, real and imaginary parts of a vector of complex numbers z
    OUTPUTS:
        z: (n,) complex, z[j] = x[2j] + i*x[2j+1]
    '''

    z = np.zeros((len(x)//2,), dtype=complex)
    z = x[np.arange(0, len(x), 2)] + 1.j*x[np.arange(1, len(x), 2)]
    return z

# Conversion from c (complex) to cc (floats)
def Comp2Real(z):
    '''
    Given a vector z of length (n,) of complex numbers, return a real array x where
    z[j] = x[2j] + i*x[2j+1]
    '''

    x = np.zeros((2*len(z),), dtype=float)
    x[np.arange(0, len(x), 2)] = np.real(z)
    x[np.arange(1, len(x), 2)] = np.imag(z)
    return x

# Uses previous runs (with fewer terms in the truncated Fourier series) to create a better starting guess for the optimisation problem than the constant function.
def BetterStartGuess(M, n, info):
    '''
    Given information about the approximation to the eigenfunctions in info, which have M=M1,
    use this to create a good starting estimate for the solve of varphi_n with M=M.
    INPUTS:
        M: int, current number of Fourier modes being used in approximation.
        n: int, (n+1) is the index of the current varphi_n to find good starting guess for.
        info: np.load, extracted information from previous runs as output by SaveRun
    OUTPUTS:
        cc0: (2(2M+1)**2,) float, the starting guess to use based off the previous runs.
    '''

    # get coefficients in lower-dimensional solve
    cVecOld = info['cVecs'][n,:]
    MOld = info['M']
    cMatOld = cVecOld.reshape((2*MOld+1, 2*MOld+1))
    cMat = np.zeros((2*M+1, 2*M+1), dtype=complex)
    # insert cMatOld into "middle" of cMat
    padding = (2*M+1 - (2*MOld+1)) // 2
    cMat[padding:padding+(2*MOld+1), padding:padding+(2*MOld+1)] = cMatOld
    # reshape into vector for output
    cVec = cMat.reshape(((2*M+1)**2,))
    cc0 = Comp2Real(cVec)

    return cc0

#%% FourierFunction class

class FourierFunction:
    '''
    ATRIBUTES:
        theta
        omega
        M
        lbda
        cMat
        cVec
    METHODS:
        val
        norm
        boundaryNorm
        set_lambda
    '''

    def __init__(self, theta, omega, c):
        '''
        Create an instance with the Fourier coefficients passed in the matrix or vector c.
        INPUTS:
            theta: (2,) float, value of the quasimomentum.
            omega: float, value of omega.
            c: (2M+1,2M+1) complex double, the Fourier coefficients of this function.
            If c is of shape  ((2M+1)^2,), then this is interpretted column-wise, as above.
        OUTPUTS:
            FourierFunction instance, with theta, omega, M, cMat, cVec, and lbda all set.
        '''

        self.theta = theta
        self.omega = omega
        if c.ndim==2:
            # passed cMat
            self.cMat = c
            self.M = int ( (c.shape[0]-1) // 2 ) #-1 is actually superflous here, but it doesn't hurt...
            self.cVec = c.reshape((c.shape[0]*c.shape[0],)) #create a view rather than a whole new matrix
        elif c.ndim==1:
            # passed cVec
            self.cVec = c
            self.M = int( (np.sqrt(c.shape[0]) - 1) // 2 ) #-1 is actually superflous here, but it doesn't hurt
            self.cMat = c.reshape((2*self.M+1,2*self.M+1))
        else:
            raise ValueError('Unexpected number of dimensions in c array, got %d, expected 1 or 2' % c.ndim)
        self.set_lambda()
        return

    def __str__(self):
        '''
        Default print output when instance of class is passed to print()
        '''
        rFig, iFig = self.plot()
        rFig[0].show()
        iFig[0].show()
        return 'FourierFunction with M = %d' % self.M

    def val(self, x):
        '''
        Evaluates the FourierFunction at the point(s) x
        INPUTS:
            x: (l,2) float, (x,y) coordinate pairs, stored in each column of the array.
        OUTPUTS:
            xVals: (l,) complex, values of the FourierFunction at the input x.
        '''

        mInts = np.arange(-self.M, self.M+1)
        if x.ndim==2:
            xVals = np.zeros(x.shape[0], dtype=complex)
            # it's a for loop because I can't wrap 3D vectorisation around my head
            for l in range(x.shape[0]):
                expAX = exp(2.j*pi*x[l,0]*mInts)
                expBY = exp(2.j*pi*x[l,1]*mInts)
                # expAX[i] = e^{2i\pi x[l]*(i-M)}, similarly for expBY
                expMat = np.outer(expAX, expBY)
                # np.outer(a,b) returns M_{i,j} = a_i * b_j, thus
                # expMat[i,j] = e^{2i\pi x[l]*(i-M)} * e^{2i\pi y[l]*(j-M)}
                # now we multiply by the Fourier coefficients...
                fTerms = expMat * self.cMat
                # then sum all the terms in the matrix!
                xVals[l] = np.copy(fTerms.sum())
        elif x.ndim==1:
            expAX = exp(2.j*pi*x[0]*mInts)
            expBY = exp(2.j*pi*x[1]*mInts)
            # expAX[i] = e^{2i\pi x[l]*(i-M)}, similarly for expBY
            expMat = np.outer(expAX, expBY)
            # np.outer(a,b) returns M_{i,j} = a_i * b_j, thus
            # expMat[i,j] = e^{2i\pi x[l]*(i-M)} * e^{2i\pi y[l]*(j-M)}
            # now we multiply by the Fourier coefficients...
            fTerms = expMat * self.cMat
            # then sum all the terms in the matrix!
            xVals = fTerms.sum()
        else:
            raise ValueError('Unexpected dimension of x, got %d, expected 1 or 2', x.ndim)
        return xVals

    def norm(self):
        '''
        Computes the L2(Omega) norm of the function.
        INPUTS:

        OUTPUTS:
            nVal: float, value of the L2(Omega) norm of this function.
        '''

        return np.sqrt( np.sum( np.abs( self.cVec )**2 ) )

    def boundaryNorm(self):
        '''
        Computes the L2(\partial Omega) norm of the function.
        INPUTS:

        OUTPUTS:
            nVal: float, value of the boundary-norm of this function.
        '''

        matProd = np.matmul(self.cMat, np.conjugate(self.cMat).T) + np.matmul(self.cMat.T, np.conjugate(self.cMat))
        sumValue = np.sum( np.real( matProd ) )
        nVal = np.sqrt( 2.*sumValue )

        return  nVal

    def ip(self, f):
        '''
        Evaulates the L2(\Omega) inner product of the FourierFunction with another instance f of the class
        INPUTS:
            f: FourierFunction, function to take inner product with
        OUTPUTS:
            inP: complex, value of the inner product <self, f>.
        '''

        return np.sum( self.cVec * np.conjugate(f.cVec) )

    def plot(self, N=250, levels=15):
        '''
        Creates a heatmap of the function's real and imaginary parts, returning the figure handles for each.
        INPUTS:
            N: int, number of meshpoints to use for domain
            levels: int, number of contour levels to use
        OUTPUTS:
            rF, aF: figure handles, handles for heatmaps of the function over the region Omega.
        '''

        rF, rAx = plt.subplots()
        iF, iAx = plt.subplots()
        for a in [rAx, iAx]:
            a.set_aspect('equal')
            a.set_xlabel(r'$x_1$')
            a.set_ylabel(r'$x_2$')
        rAx.set_title(r'$\Re(\varphi)$, $\lambda=%.3f $' % (self.lbda))
        iAx.set_title(r'$\Im(\varphi)$, $\lambda=%.3f $' % (self.lbda))

        X = np.linspace(0,1,num=N)
        u = np.zeros((N,N),dtype=complex)
        for i,x in enumerate(X):
            for j,y in enumerate(X):
                u[j,i] = self.val(np.asarray([x,y]))
        rCon = rAx.contourf(X, X, np.real(u), levels=levels)
        iCon = iAx.contourf(X, X, np.imag(u), levels=levels)
        # make colourbars
        rF.colorbar(rCon)
        iF.colorbar(iCon)
        return [rF, rAx], [iF, iAx]

    def set_lambda(self):
        '''
        Sets the value of lambda to be that of the functional J(c), using c = self.cVec.
        If we have solved the corresponding minimisation problem for the vector self.cVec, this will set lambda to
        be the eigenvalue corresponding to this eigenfunction.
        '''
        self.lbda = J(self.cVec, self.theta, self.omega)
        return

#%% Functions that pertain to the objective functional

# Objective functional
def J(c, theta, omega):
    '''
    Evaluates the objective function J.
    INPUTS:
        c: (2M+1)^2 complex, vector of the c_ab^n (\mathbf{c} above)
        theta: (2,) float, value of the quasi-momentum
        omega: float, value of omega
    OUTPUTS:
        Jval: float, value of the objective function J
    '''

    M = (np.sqrt(c.shape[0]) - 1) // 2
    sqCoeffs = np.abs(c)*np.abs(c)

    alpha = beta = np.arange(-M, M+1)
    # this is a ((2M+1)**2,2) numpy array of all combinations of alpha, beta that we need to use,
    # these are stacked by [a0, b0], [a0, b1], ..., [a0, B(2M+1)], [a1, b0] etc, IE as \mathbf{c} is.
    abVals = np.array(np.meshgrid(alpha, beta)).T.reshape(-1,2)
    # now compute the theta - 2\pi (a,b) values.... again as an ((2M+1)**2,2) array
    tMinusIndex = theta.reshape((1,2)) - 2*pi*abVals
    # now compute |theta - 2\pi (a,b)|^2 - \omega^2, as an ((2M+1)**2,) array
    prods = np.linalg.norm(tMinusIndex, axis=1)**2 - omega*omega
    # it should now just be a case of a sum of element-wise vector products
    Jval = np.sum(sqCoeffs * prods)
    return Jval

# Evaluates the Jacobian of the objective functional as a function of cc, rather than c
def J_Jac(cc, theta, omega):
    '''
    Evaluates the Jacobian of the objective function J that takes cc = Comp2Real(c) as an input.
    INPUTS:
        cc: (2(2M+1)^2,) float, vector of the c_ab^n (\mathbf{c} above) passed through Comp2Real
        theta: (2,) float, value of the quasi-momentum
        omega: float, value of omega
    OUTPUTS:
        jacVal: (2(2M+1)^2,) float, value of the Jacobian of the objective function J
    '''

    M = (np.sqrt(cc.shape[0]//2) - 1) // 2
    alpha = beta = np.arange(-M, M+1)
    # this is a ((2M+1)**2,2) numpy array of all combinations of alpha, beta that we need to use,
    # these are stacked by [a0, b0], [a0, b1], ..., [a0, B(2M+1)], [a1, b0] etc, IE as \mathbf{c} is.
    abVals = np.array(np.meshgrid(alpha, beta)).T.reshape(-1,2)
    # now compute the theta - 2\pi (a,b) values.... again as an ((2M+1)**2,2) array
    tMinusIndex = theta.reshape((1,2)) - 2*pi*abVals
    # now compute |theta - 2\pi (a,b)|^2 - \omega^2, as an ((2M+1)**2,) array
    prods = np.linalg.norm(tMinusIndex, axis=1)**2 - omega*omega
    # now we realise that jacVal[0::2] = 2 * cc[0::2] * prods (element-wise), and
    # jacVal[1::2] = 2 * cc[1::2] * prods
    jacVal = np.zeros_like(cc, dtype=float)
    jacVal[0::2] = 2. * cc[0::2] * prods
    jacVal[1::2] = 2. * cc[1::2] * prods
    return jacVal

# Evaulates the Hessian of J as a function of cc, rather than c
def J_Hess(cc, theta, omega):
    '''
    Evaulates the Hessian of the objective function J as a function of cc  Comp2Real(c).
    INPUTS:
        cc: (2(2M+1)^2,) float, vector of the c_ab^n (\mathbf{c} above) passed through Comp2Real
        theta: (2,) float, value of the quasi-momentum
        omega: float, value of omega
    OUTPUTS:
        hessVal: (2(2M+1)^2,2(2M+1)^2) float, value of the Hessian of the objective function J
    '''

    M = (np.sqrt(cc.shape[0]//2) - 1) // 2

    alpha = beta = np.arange(-M, M+1)
    # this is a ((2M+1)**2,2) numpy array of all combinations of alpha, beta that we need to use,
    # these are stacked by [a0, b0], [a0, b1], ..., [a0, B(2M+1)], [a1, b0] etc, IE as \mathbf{c} is.
    abVals = np.array(np.meshgrid(alpha, beta)).T.reshape(-1,2)
    # now compute the theta - 2\pi (a,b) values.... again as an ((2M+1)**2,2) array
    tMinusIndex = theta.reshape((1,2)) - 2*pi*abVals
    # now compute |theta - 2\pi (a,b)|^2 - \omega^2, as an ((2M+1)**2,) array
    prods = np.linalg.norm(tMinusIndex, axis=1)**2 - omega*omega
    # now we just notice that H[1::2,1::2] = H[0::2,0::2] = diag(prods)
    hessVal = np.zeros((2*(2*int(M)+1)**2,2*(2*int(M)+1)**2), dtype=float)
    hessVal[0::2,0::2] = np.diag(prods)
    hessVal[1::2,1::2] = np.diag(prods)
    return hessVal

# Evaulates both the objective function and it's Jacobian as functions of cc.
def J_PlusJac(cc, theta, omega):
    '''
    Evalutes J and it's Jacobian matrix, returning the pair as a tuple. This should be marginally faster than
    computing both individually.
    Note that the input is cc, not c!
    INPUTS:
        cc: (2(2M+1)^2,) float, vector of the c_ab^n (\mathbf{c} above) passed through Comp2Real
        theta: (2,) float, value of the quasi-momentum
        omega: float, value of omega
    OUTPUTS:
        jVal: float, value of the objective function J
        jacVal: (2(2M+1)^2,) float, value of the Jacobian of J
    '''

    M = (np.sqrt(cc.shape[0]//2) - 1) // 2
    sqCoeffs = np.abs(Real2Comp(cc))**2

    alpha = beta = np.arange(-M, M+1)
    # this is a ((2M+1)**2,2) numpy array of all combinations of alpha, beta that we need to use,
    # these are stacked by [a0, b0], [a0, b1], ..., [a0, B(2M+1)], [a1, b0] etc, IE as \mathbf{c} is.
    abVals = np.array(np.meshgrid(alpha, beta)).T.reshape(-1,2)
    # now compute the theta - 2\pi (a,b) values.... again as an ((2M+1)**2,2) array
    tMinusIndex = theta.reshape((1,2)) - 2*pi*abVals
    # now compute |theta - 2\pi (a,b)|^2 - \omega^2, as an ((2M+1)**2,) array
    prods = np.linalg.norm(tMinusIndex, axis=1)**2 - omega*omega
    # it should now just be a case of a sum of element-wise vector products
    jVal = np.sum(sqCoeffs * prods)
    # now we realise that jacVal[0::2] = 2 * cc[0::2] * prods (element-wise), and
    # jacVal[1::2] = 2 * cc[1::2] * prods
    jacVal = np.zeros_like(cc, dtype=float)
    jacVal[0::2] = 2. * cc[0::2] * prods
    jacVal[1::2] = 2. * cc[1::2] * prods
    return jVal, jacVal

#%% Pertaining to the constraints on the optimisiation problem

# Computes the first row of the Jacobian of the constraints vector
def FirstJacRow(cc):
    '''
    Builds the Jacobian row corresponding to the norm constraint on the boundary
    '''
    J = cc.shape[0]
    M = int((np.sqrt(J//2) - 1) // 2)

    matMeth = np.zeros((J,),dtype=float)
    cMatReal = cc[0::2].reshape((2*M+1,2*M+1)) #this is real(cMat), essentially
    cMatImag = cc[1::2].reshape((2*M+1,2*M+1)) #this is imag(cMat), essentially
    # even indexed j
    for j in range(0,J,2):
        beta = (j//2) % (2*M+1) #gets beta from j/2 = beta + (2M+1)*alpha
        alpha = (j//2) // (2*M+1) #gets alpha from j/2 = beta + (2M+1)*alpha
        matMeth[j] = 2.* ( np.sum(cMatReal[alpha,:]) + np.sum(cMatReal[:,beta]) )
    # odd indexed j
    for j in range(1,J,2):
        #odd j's, so imag parts here
        beta = (j//2) % (2*M+1) #gets beta from j/2 = beta + (2M+1)*alpha
        alpha = (j//2) // (2*M+1) #gets alpha from j/2 = beta + (2M+1)*alpha
        matMeth[j] = 2.* ( np.sum(cMatImag[alpha,:]) + np.sum(cMatImag[:,beta]) )

    return matMeth

# Builds the Hessian of the vector of constraints dotted with the vector of Lagrange multipliers
def BuildHessian(cc):
    '''
    Builds the constant Hessian matrix for the norm-boundary constraint
    '''
    J = cc.shape[0]
    M = int((np.sqrt(J//2) - 1) // 2)
    H = np.zeros((J,J), dtype=float)
    Hsmall = np.zeros((J//2,J//2), dtype=float)
    for i in range(J//2):
        beta = i % (2*M+1) #gets beta from i/2 = beta + (2M+1)*alpha
        alpha = i // (2*M+1) #gets alpha from i/2 = beta + (2M+1)*alpha
        for j in range(J//2):
            q = j % (2*M+1) # j/2 = q + (2M+1)p
            p = j // (2*M+1) # j/2 = q + (2M+1)p
            Hsmall[i,j] = 2. * ( KronDelta(alpha,p) + KronDelta(beta,q) )
    H[0::2,0::2] = Hsmall
    H[1::2,1::2] = Hsmall
    return H

# Kronecker delta function
def KronDelta(i,j):
    '''
    Kronecker delta: 1 if i==j, 0 otherwise.
    '''
    if i==j:
        return 1
    else:
        return 0

# Constructs functions which evaulate the constraints vector, it's Jacobian, and it's Hessian
def BuildConstraints(n, prevPhis=[]):
    '''
    Constructs the functions `fun(cc)`, `Jac(cc)`, and `Hess(cc,x)` for the minimisation problem to find \varphi_n.
    Also pass back the vector defining the equalities that must be satisfied.
    INPUTS:
        n: int, we will be solving for \varphi_n.
        prevPhis: (n-1) list, a list of FourierFunction objects containing the representations of the functions
        \varphi_1 through \varphi_(n-1), which are required for the orthogonality constraints.
    OUTPUTS:
        Fun: lambda cc: the function fun(cc) above, taking cc as input and returning shape (2*n-1,).
        Jac: lambda cc: the function Jac(cc) above, taking cc as input and returning shape (2*n-1, len(cc)).
        Hess: lambda (cc,v): the function Hess(cc,v) described above, taking (cc,v)
        as input and returning shape (len(cc),len(cc))
        equalityVector: (2*n-1,) float, the zero vector except with index 0 element set to 1/4
        for the boundary norm constraint.
    '''

    if n<=0:
        raise ValueError('n <= 0, varphi indexing starts from 0')
    elif n-len(prevPhis)!=1:
        raise ValueError('Unexpected number of previous functions provided (got %d, expected %d)' \
                         % (len(prevPhis),n-1))


    def Fun(cc):
        '''
        Takes cc as argument, and returns the vector of constraint expressions.
        '''

        J = cc.shape[0]
        M = int((np.sqrt(J//2) - 1) // 2)
        cMat = Real2Comp(cc).reshape((2*M+1,2*M+1))
        constraints = np.zeros((2*n-1,), dtype=float)
        constraints[0] = np.sum(np.real( \
                                        np.matmul( cMat, np.conjugate(cMat).T ) \
                                        + np.matmul( cMat.T, np.conjugate(cMat) ) \
                                       ))
        # now fill in orthogonality constraints, if there are any
        # NB: range(start, stop, step)
        for i in range(1,2*n-1,2):
            # cc[0::2]*cc_k[0::2] + cc[1::2]*cc_k[1::2], where 0<i is odd, and k = (i+1)//2
            k = (i+1)//2
            cc_k = Comp2Real(prevPhis[k-1].cVec) #note -1 since varphi_k is stored in prevPhis[k-1]
            constraints[i] = np.sum( cc[0::2]*cc_k[0::2] + cc[1::2]*cc_k[1::2] )
        for i in range (2,2*n-1,2):
            # cc[1::2]*cc_k[0::2] - cc[0::2]*cc_k[1::2], where 0<i is even, and k = i//2
            k = i//2
            cc_k = Comp2Real(prevPhis[k-1].cVec) #again, -1 due to offsets
            constraints[i] = np.sum( cc[1::2]*cc_k[0::2] - cc[0::2]*cc_k[1::2] )
        return constraints

    def Jac(cc):
        '''
        Takes cc as argument, and returns the Jacobian of Fun(cc)
        '''

        jac = np.zeros(((2*n-1), len(cc)), dtype=float)
        jac[0,:] = FirstJacRow(cc)
        # now the rows that come from the orthogonality constraints
        for i in range(1,2*n-1,2):
            k = (i+1)//2
            cc_k = Comp2Real(prevPhis[k-1].cVec)
            jac[i,:] = cc_k
        for i in range(2,2*n-1,2):
            k = i//2
            cc_k = Comp2Real(prevPhis[k-1].cVec)
            pows = ( -1*np.ones_like(cc_k, dtype=int) ) ** np.arange(len(cc_k))
            inds = np.arange(len(cc_k)) + pows
            jac[i,:] = -pows * cc_k[inds]
        return jac

    def Hess(cc, v):
        '''
        Takes cc and the array of Lagrange multipliers v as arguments, returning the Hessian of dot(fun(cc), v).
        '''
        return BuildHessian(cc) * v[0]

    equalityVector = np.zeros((2*n-1,), dtype=float)
    equalityVector[0] = 0.5 # this is the (square of the) norm on the boundary constraint, all others are zero

    return Fun, Jac, Hess, equalityVector

#%% Command-line executions

if __name__=='__main__':

	parser = argparse.ArgumentParser(description='Approximation of eigenvalues and eigenfunctions for the Dirichlet-to-Neumann map, for the Cross-in-plane geometry.')
	parser.add_argument('-M', type=int, help='<Required> Order of the highest of Fourier mode to truncate series at.', required=True)
	parser.add_argument('-N', type=int, help='<Required> Number of eigenfunctions and eigenvalues to find (starting from lowest eigenvalue)', required=True)
	parser.add_argument('-omega', type=float, help='<Required> Value of omega in objective function', required=True)
	parser.add_argument('-fOut', default='', type=str, help='Output file name, if blank filename will be auto-generated')
	parser.add_argument('-fDump', default='./DtNEfuncs/', type=str, help='Directory to save output file to, default is ./')
	parser.add_argument('-t1', default=0.0, type=float, help='QM_1 will be set to this value multiplied by pi.')
	parser.add_argument('-t2', default=0.0, type=float, help='QM_2 will be set to this value multiplied by pi.')
	parser.add_argument('-nIts', default=1000, type=int, help='Maximum number of iterations for solver')
	parser.add_argument('-noJH', action='store_true', help='Solver will not use analytic Jacobian or Hessian')
	parser.add_argument('-prevInfo', default='', type=str, help='Path to file storing information that is to be used for initial guesses in this run')
	parser.add_argument('-lOff', action='store_true', help='Suppress printing of progress and solver log to the screen')

	# extract input arguments and get the setup ready
	args = parser.parse_args()

	# There will be (2M+1)^2 Fourier modes, the "highest" being of order M
	M = args.M
	# We want to find this many eigenfunctions and eigenvectors
	N = args.N
	if (2*M+1)**2 < N:
	    raise ValueError('(2M+1)^2 (%d) < (%d) N: cannot find orthogonal functions. ABORT' % ((2*M+1)**2, N))
	else:
		print('Will find %d eigenpairs, approximating with %d-dimensional Fourier space' % (N, (2*M+1)**2))

	# quasi-momentum value
	theta = np.array([args.t1, args.t2], dtype=float) * pi
	# value of omega
	omega = args.omega
	print('Read omega = %.3f and theta = [%.3f, %.3f]\pi' % (omega, args.t1, args.t2))

	# create solver options handle
	options = {'maxiter' : args.nIts, 'disp' : (not args.lOff) }

	# create store for eigenfunctions
	phiList = []

	# check to see if previous information was provided
	# first, establish the default behaviour
	nPrev = 0
	nStart = 0
	if args.prevInfo:
		# empty string evaluates to false, so if previous run information was provided, we get to here
		# load previous run information
		prevInfo = np.load(args.prevInfo)
		# how many previous functions are provided?
		# If we we want more than before, we'll still need the ignorant starting guess
		# cVecs are stored row-wise, so number of rows = number of previous guesses available
		nPrev = np.shape(prevInfo['cVecs'])[0]
		print('Found previous information file: ' + args.prevInfo + ', containing %d eigenfunctions' % nPrev)
		# note: if the previous M was the same, we don't need to bother recomputing the N functions we've been given!
		if prevInfo['M'] == M:
			# don't bother recomputing stuff for those N that we already have!
			print('Supplied first %d eigenfunctions for M = %d.' % (nPrev, M))
			print('Will only search for varphi_%d through varphi_%d.' % (nPrev+1, N))
			for n in range(nPrev):
				cc0 = BetterStartGuess(M, n, prevInfo)
				phiList.append( FourierFunction(prevInfo['theta'], prevInfo['omega'], prevInfo['cVecs'][n,:]) )
			# then, inform the loops that we can start computing from n=nStart, rather than n=0
			nStart = nPrev

	# If there is no previous information given, or we want more e'funcs than prevInfo provides,
	# we'll have to use the default starting guess: the constant function with boundary norm 1
	cc0Def = np.zeros(((2*M+1)**2,), dtype=complex)
	cc0Def[2*(M+1)*(M)] += 0.5
	cc0Def = Comp2Real(cc0Def)
	
	# flag failed convergence if it happens
	noConv = False
	
    # all options are setup, commence the solve
	if (not args.noJH) and (not args.lOff):
		# use the analytic Jacobian and Hessian, and print the log to the screen
		print('Beginning solve: using analytic Jacobian & Hessian')

		# returns the functional J and it's Jacobian as a tuple
		Jopt = lambda cc: J_PlusJac(cc, theta, omega)
		# find each eigenfunction in sequence
		for n in range(nStart, N):
			print(' ----- \n [%d] Start time:' % (n+1), end=' ')
			print(datetime.now().strftime("%H:%M"))

			# setup constraints
			Fun_n, Jac_n, Hess_n, eqV_n = BuildConstraints(n+1, prevPhis=phiList)
			min_constraints = NonlinearConstraint(Fun_n, eqV_n, eqV_n, jac=Jac_n, hess=Hess_n)

			# starting guess update if better information was provided
			if n < nPrev:
				cc0 = BetterStartGuess(M, n, prevInfo)
				fPrev = FourierFunction(prevInfo['theta'], prevInfo['omega'], prevInfo['cVecs'][n,:])
			else:
				# no previous information (either not provided or out of previous functions)
				# try using the default starting guess
				cc0 = cc0Def

			t0 = time.time()
			# SLSQP doesn't use the Hessian btw, so we just don't pass it in
			result = minimize(Jopt, cc0, jac=True, constraints=min_constraints, \
					 options=options, method='SLSQP')
			t1 = time.time()

			# construct answer as FourierFunction
			cVec = Real2Comp(result.x)
			f = FourierFunction(theta, omega, cVec)
			phiList.append(f)

			# print log info
			print(' Boundary norm of varphi_%d: %.2e' % (n+1, f.boundaryNorm()))
			if n < nPrev:
				print(' \lambda_%d: %.5f     -     Change from previous: %.3e' % (n+1, f.lbda, fPrev.lbda - f.lbda))
			else:
				print(' \lambda_%d: %.5f' % (n+1, f.lbda))
			print(' Runtime: approx %d mins (%s seconds)' % (np.round((t1-t0)/60), t1-t0))

			if result.status!=0:
				# this run failed to converge, break loop
				print(' Failed to find eigenfunction, terminate loop early')
				noConv = True
				break

			print(' ----- ')
	elif (not args.noJH):
		# use the analytic Jacobian and Hessian, do not print log to screen

		# returns the functional J and it's Jacobian as a tuple
		Jopt = lambda cc: J_PlusJac(cc, theta, omega)
		# find each eigenfunction in sequence
		for n in range(nStart, N):
			# setup constraints
			Fun_n, Jac_n, Hess_n, eqV_n = BuildConstraints(n+1, prevPhis=phiList)
			min_constraints = NonlinearConstraint(Fun_n, eqV_n, eqV_n, jac=Jac_n, hess=Hess_n)

			# starting guess update if better information was provided
			if n < nPrev:
				cc0 = BetterStartGuess(M, n, prevInfo)
				fPrev = FourierFunction(prevInfo['theta'], prevInfo['omega'], prevInfo['cVecs'][n,:])
			else:
				# no previous information (either not provided or out of previous functions)
				# try using the default starting guess
				cc0 = cc0Def

			t0 = time.time()
			# SLSQP doesn't use the Hessian btw, so we just don't pass it in
			result = minimize(Jopt, cc0, jac=True, constraints=min_constraints, \
					 options=options, method='SLSQP')
			t1 = time.time()

			# construct answer as FourierFunction
			cVec = Real2Comp(result.x)
			f = FourierFunction(theta, omega, cVec)
			phiList.append(f)

			if result.status!=0:
				# this run failed to converge, break loop
				print(' Failed to find eigenfunction [n=%d], terminate loop early' % (n+1))
				noConv = True
				break
	elif (not args.lOff):
		# do not use analytic Jacobian and Hessian, but do print to the screen
		print('Beginning solve: NOT USING analytic Jacobian & Hessian')

		# returns the functional J
		Jopt = lambda cc: J(Real2Comp(cc), theta, omega)
		# find each eigenfunction in sequence
		for n in range(nStart, N):
			print(' ----- \n [%d] Start time:' % (n+1), end=' ')
			print(datetime.now().strftime("%H:%M"))

			# setup constraints
			Fun_n, Jac_n, Hess_n, eqV_n = BuildConstraints(n+1, prevPhis=phiList)
			min_constraints = NonlinearConstraint(Fun_n, eqV_n, eqV_n, jac=Jac_n, hess=Hess_n)

			# starting guess update if better information was provided
			if n < nPrev:
				cc0 = BetterStartGuess(M, n, prevInfo)
				fPrev = FourierFunction(prevInfo['theta'], prevInfo['omega'], prevInfo['cVecs'][n,:])
			else:
				# no previous information (either not provided or out of previous functions)
				# try using the default starting guess
				cc0 = cc0Def

			t0 = time.time()
			# SLSQP doesn't use the Hessian btw, so we just don't pass it in
			result = minimize(Jopt, cc0, constraints=min_constraints, \
					 options=options, method='SLSQP')
			t1 = time.time()

			# construct answer as FourierFunction
			cVec = Real2Comp(result.x)
			f = FourierFunction(theta, omega, cVec)
			phiList.append(f)

			# print log info
			print(' Boundary norm of varphi_%d: %.2e' % (n+1, f.boundaryNorm()))
			if n < nPrev:
				print(' \lambda_%d: %.5f     -     Change from previous: %.3e' % (n+1, f.lbda, fPrev.lbda - f.lbda))
			else:
				print(' \lambda_%d: %.5f' % (n+1, f.lbda))
			print(' Runtime: approx %d mins (%s seconds)' % (np.round((t1-t0)/60), t1-t0))

			if result.status!=0:
				# this run failed to converge, break loop
				print(' Failed to find eigenfunction, terminate loop early')
				noConv = True
				break

			print(' ----- ')
	else:
		# do not use analytic Jacobian and Hessian, do not print to screen

		# returns the functional J
		Jopt = lambda cc: J(Real2Comp(cc), theta, omega)
		# find each eigenfunction in sequence
		for n in range(nStart, N):
			# setup constraints
			Fun_n, Jac_n, Hess_n, eqV_n = BuildConstraints(n+1, prevPhis=phiList)
			min_constraints = NonlinearConstraint(Fun_n, eqV_n, eqV_n, jac=Jac_n, hess=Hess_n)

			# starting guess update if better information was provided
			if n < nPrev:
				cc0 = BetterStartGuess(M, n, prevInfo)
				fPrev = FourierFunction(prevInfo['theta'], prevInfo['omega'], prevInfo['cVecs'][n,:])
			else:
				# no previous information (either not provided or out of previous functions)
				# try using the default starting guess
				cc0 = cc0Def

			t0 = time.time()
			# SLSQP doesn't use the Hessian btw, so we just don't pass it in
			result = minimize(Jopt, cc0, constraints=min_constraints, \
					 options=options, method='SLSQP')
			t1 = time.time()

			# construct answer as FourierFunction
			cVec = Real2Comp(result.x)
			f = FourierFunction(theta, omega, cVec)
			phiList.append(f)

			if result.status!=0:
				# this run failed to converge, break loop
				print(' Failed to find eigenfunction, terminate loop early')
				noConv = True
				break

	# run is now complete - save the output if we converged!
	if (not noConv):
		infoFile = SaveRun(theta, omega, phiList, fname=args.fOut, fileDump=args.fDump)
		# inform the user of completion of the script
		print('Completed, output file saved to:' + infoFile)
		with open('DtN-Minimiser-ConveyerFile.txt', 'w') as fh:
			fh.write("%s" % infoFile)
			sys.exit(0)
	else:
		sys.exit(1)
