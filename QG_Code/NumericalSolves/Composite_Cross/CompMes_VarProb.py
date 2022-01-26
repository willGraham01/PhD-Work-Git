#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 12:04:28 2021

@author: will

This file allows the content of CompositeMeasure-VariationalProblem.ipynb to be called as a script from the command line.
For documentation, see the aforementioned file.
"""
# sys.argv[1:] contains command line arguments. Index 0 is just the script name
import argparse
# for returning values to the command line
import sys

import numpy as np
from numpy import pi

from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from scipy.optimize import basinhopping

import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

import time
from datetime import datetime

#%% Utility functions; conversion between real and complex, division overide, SaveRun

def DivOveride(a, b):
    '''
    Performs our "override division" operation a/b, where we interpret division by 0 as returning 0.
    INPUTS:
        a: (n,), numerator array or scalar
        b: (n,), denominator array or scalar
    OUTPUTS:
        quo: (n,) float, result of division a/b and setting division by 0 to result in 0
    '''
    return np.divide(a, b, out=np.zeros_like(b, dtype=float), where=b!=0)

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

def Comp2Real(z):
    '''
    Given a vector z of length (n,) of complex numbers, return a real array x where
    z[j] = x[2j] + i*x[2j+1]
    '''
    
    x = np.zeros((2*len(z),), dtype=float)
    x[np.arange(0, len(x), 2)] = np.real(z)
    x[np.arange(1, len(x), 2)] = np.imag(z)
    return x

def SaveRun(theta, uRealStore, fname='', fileDump='./Poly2D_Results/'):
    '''
    Saves the output of the minimisation proceedure (the coefficients) to a .npz file, 
    which can then be read in by another script or notebook.
    INPUTS:
        theta: (2,) float, value of the quasi-momentum.
        uRealStore: (n,2M^2) float, the coefficients of the Poly2D functions (in real format) found by the optimisation
        fname: str, the name of the file to which values will be saved.
        fileDump: str, path to the directory in which to save the .npz file.
    OUTPUTS:
        outFile: str, name of file to which values were saved
    '''
    
    M = np.sqrt( np.shape(uRealStore)[1] // 2 )
    
    if not fname:
        #automatically create file name
        timeStamp = datetime.now()
        outFile = fileDump + timeStamp.strftime("%Y-%m-%d_%H-%M_PolyCoeffsM-") + str(int(M)) + '.npz'
    else:
        outFile = fileDump + fname
    
    np.savez(outFile, theta=theta, uRealStore=uRealStore)
    return outFile

#%% Basis vectors; inner product matrices assembler functions

def Lambda2Stores(M, theta):
    '''
    Builds matrices that will store the various inner products wrt lambda_2 between the basis functions, 
    so that our later computations do not have to.
    INPUTS:
        M: int, M-1 is the highest order of the polynomial basis we are using
        theta: (2,) float, value of the quasi-momentum
    OUTPUTS: 
    All are shape (M^2-1,M^2-1) and list inner products wrt lambda_2 only.
        pmpn: float, pmpn[m,n] = p+m*p_n
        l2mn: float, l2mn[m,n] = <phi_m,phi_n> / pmpn
        l2mdn: float, l2mdn[m,n] = <i\theta phi_m, \grad phi_n> / (pmpn*1.j)
        l2dmdn: float, l2dmdn[m,n] = <\grad phi_m, \grad phi_n> / pmpn
        l2dtmdtn: complex, l2dtmdtn[m,n] = <\grad^\theta phi_m, \grad^\theta phi_n> / pmpn
    '''

    # SETUP: Create array of combinations of i & j values for a given m
    
    iInds = jInds = np.arange(0,M,dtype=int)
    ijVals = np.array(np.meshgrid(iInds, jInds)).T.reshape(-1,2) # ijVals[m,:] = [i_m,j_m]
    # np.add.outer(a,b) returns an array AB with entries AB[m,n] = a[m]+b[n]
    Imn = np.add.outer(ijVals[:,0],ijVals[:,0]) # Imn[m,n] = i_m+i_n
    Jmn = np.add.outer(ijVals[:,1],ijVals[:,1]) # Jmn[m,n] = j_m+j_n
    
    # first, we build a vector of the p_m
    # p_m = \sqrt{ \frac{(2i_m+1)(2j_m+1)}{2i_m + 2j_m +3} }
    pm = (2*ijVals[:,0] + 1)*(2*ijVals[:,1]+1) / ( 2*ijVals[:,0] + 2*ijVals[:,1] + 3 )
    pm = np.sqrt(pm)
    
    # next, we need to compute the various inner products
    # we'll store these in a matrix of shape (M^2-1,M^2-1,l) where
    # l is the number of inner products that we want to store
    # details will be given below
    
    pmpn = np.outer(pm,pm)
    # pmpn[m,n] = pm * pn
    # this also saves us from appending pmpn to all our following calculations!
    
    # stores at index (m,n) the value <phi_m,phi_n>_{lambda2} / pmpn
    # note: l2mn is symmetric!
    l2mn = (1/(Imn+1)) * (1/(Jmn+1))
    
    # stores at index (m,n) the value <i\theta phi_m, \grad phi_n>_{lambda2} / (pmpn*1.j)
    # NOTE: the decision to leave out the i and store this as a float to save memory space!
    l2mdn = theta[0]*(DivOveride(1,Imn)*ijVals[:,0])/(Jmn+1) + theta[1]*(DivOveride(1,Jmn)*ijVals[:,1])/(Imn+1)
    
    # stores at index (m,n) the value <\grad phi_m, \grad phi_n>_{lambda2} / pmpn
    # note: l2dmdn is symmetric!
    l2dmdn = DivOveride( np.outer(ijVals[:,0], ijVals[:,0]), Imn-1 ) * (1/(Jmn+1))
    l2dmdn += DivOveride( np.outer(ijVals[:,1], ijVals[:,1]), Jmn-1 ) * (1/(Imn+1))
    
    # stores at index (m,n) the value <\grad^\theta phi_m, \grad^\theta phi_n>_{lambda2} / pmpn
    # note: l2mdn.T "=" l2ndm
    l2dtmdtn = l2dmdn + 1.j*l2mdn.T - 1.j*l2mdn + np.dot(theta, theta)*l2mn
    
    return pmpn, l2mn, l2mdn, l2dmdn, l2dtmdtn

def LambdaHStores(M, theta):
    '''
    Builds matrices that will store the various inner products wrt lambda_h between the basis functions, 
    so that our later computations do not have to.
    INPUTS:
        M: int, M-1 is the highest order of the polynomial basis we are using
        theta: (2,) float, value of the quasi-momentum
    OUTPUTS: 
    All are shape (M^2-1,M^2-1) and list inner products wrt lambda_h only.
    pmpn[m,n] = p_m * p_n throughout.
        lhmn: float, lhmn[m,n] = <phi_m,phi_n> / pmpn
        lhmdn: float, lhmdn[m,n] = <i\theta_h phi_m, \grad phi_n> / (pmpn*1.j)
        lhdmdn: float, lhdmdn[m,n] = <\grad phi_m, \grad phi_n> / pmpn
        lhdtmdtn: complex, lhdtmdtn[m,n] = <\grad^\theta phi_m, \grad^\theta phi_n> / pmpn
    '''
    
    # SETUP: Create array of combinations of i & j values for a given m
    
    iInds = jInds = np.arange(0,M,dtype=int)
    ijVals = np.array(np.meshgrid(iInds, jInds)).T.reshape(-1,2) # ijVals[m,:] = [i_m,j_m]
    # np.add.outer(a,b) returns an array AB with entries AB[m,n] = a[m]+b[n]
    Imn = np.add.outer(ijVals[:,0],ijVals[:,0]) # Imn[m,n] = i_m+i_n
    Jmn = np.add.outer(ijVals[:,1],ijVals[:,1]) # Jmn[m,n] = j_m+j_n
    
    # throughout, we only need to do these for when m = i_m M and n = i_n M
    # as all other IP's are 0 on this edge.
    # Jmn[m,n] > 0 <=> j_m or j_n != 0, so this can be used to determine where we should leave 0's
    
    # stores at index (m,n) the value <phi_m,phi_n> / pmpn
    lhmn = np.divide(1, Imn+1, out=np.zeros_like(Imn, dtype=float), where=Jmn==0)
    
    # stores at index (m,n) the value <i\theta_h phi_m, \grad phi_n> / (pmpn*1.j)
    # QM on this edge is theta[0]
    lhmdn = theta[0] * DivOveride(1, Imn) * ijVals[:,0]
    lhmdn[Jmn>0] = 0. #set the values we didn't need to look at to be 0
    
    # stores at index (m,n) the value <\grad phi_m, \grad phi_n> / pmpn
    lhdmdn = DivOveride( np.outer(ijVals[:,0], ijVals[:,0]), Imn-1 )
    lhdmdn[Jmn>0] = 0. #set the values we didn't need to look at to be 0
    
    # stores at index (m,n) the value <\grad^\theta phi_m, \grad^\theta phi_n> / pmpn
    lhdtmdtn = lhdmdn + 1.j*lhmdn.T - 1.j*lhmdn + np.dot(theta[0], theta[0])*lhmn
    
    return lhmn, lhmdn, lhdmdn, lhdtmdtn

def LambdaVStores(M, theta):
    '''
    Builds matrices that will store the various inner products wrt lambda_v between the basis functions, 
    so that our later computations do not have to.
    INPUTS:
        M: int, M-1 is the highest order of the polynomial basis we are using
        theta: (2,) float, value of the quasi-momentum
    OUTPUTS: 
    All are shape (M^2-1,M^2-1) and list inner products wrt lambda_v only.
    pmpn[m,n] = p_m * p_n throughout.
        lvmn: float, lvmn[m,n] = <phi_m,phi_n> / pmpn
        lvmdn: float, lvmdn[m,n] = <i\theta_v phi_m, \grad phi_n> / (pmpn*1.j)
        lvdmdn: float, lvdmdn[m,n] = <\grad phi_m, \grad phi_n> / pmpn
        lvdtmdtn: complex, lvdtmdtn[m,n] = <\grad^\theta phi_m, \grad^\theta phi_n> / pmpn
    '''
    
    # SETUP: Create array of combinations of i & j values for a given m
    
    iInds = jInds = np.arange(0,M,dtype=int)
    ijVals = np.array(np.meshgrid(iInds, jInds)).T.reshape(-1,2) # ijVals[m,:] = [i_m,j_m]
    # np.add.outer(a,b) returns an array AB with entries AB[m,n] = a[m]+b[n]
    Imn = np.add.outer(ijVals[:,0],ijVals[:,0]) # Imn[m,n] = i_m+i_n
    Jmn = np.add.outer(ijVals[:,1],ijVals[:,1]) # Jmn[m,n] = j_m+j_n
    
    # throughout, we only need to do these for when m = i_m M and n = i_n M
    # as all other IP's are 0 on this edge.
    # Jmn[m,n] > 0 <=> j_m or j_n != 0, so this can be used to determine where we should leave 0's
    
    # stores at index (m,n) the value <phi_m,phi_n> / pmpn
    lvmn = np.divide(1, Jmn+1, out=np.zeros_like(Jmn, dtype=float), where=Imn==0)
    
    # stores at index (m,n) the value <i\theta_h phi_m, \grad phi_n> / (pmpn*1.j)
    # QM on this edge is theta[1]
    lvmdn = theta[1] * DivOveride(1, Jmn) * ijVals[:,1]
    lvmdn[Imn>0] = 0. #set the values we didn't need to look at to be 0
    
    # stores at index (m,n) the value <\grad phi_m, \grad phi_n> / pmpn
    lvdmdn = DivOveride( np.outer(ijVals[:,1], ijVals[:,1]), Jmn-1 )
    lvdmdn[Imn>0] = 0. #set the values we didn't need to look at to be 0
    
    # stores at index (m,n) the value <\grad^\theta phi_m, \grad^\theta phi_n> / pmpn
    lvdtmdtn = lvdmdn + 1.j*lvmdn.T - 1.j*lvmdn + np.dot(theta[1], theta[1])*lvmn
    
    return lvmn, lvmdn, lvdmdn, lvdtmdtn

def TLambdaStores(M, theta):
    '''
    Computes the matrices whose (m,n)-th entries contain the inner products 
    <\grad^\theta phi_m, \grad^\theta phi_n>_{\compMes} and
    < phi_m, phi_n>_{\compMes}.
    INPUTS:
        M: int, M-1 is the highest order of the polynomial basis we are using
        theta: (2,) float, value of the quasi-momentum
    OUTPUTS:
    All are shape (M^2-1,M^2-1) and list inner products wrt \tilde{\lambda} only.
        ip: float, ip[m,n] = < phi_m, phi_n>_{\compMes}
        ipDT: complex, ipD[m,n] =  <\grad^\theta phi_m, \grad^\theta phi_n>_{\compMes}
    '''
    
    # bulk terms
    pmpn, l2mn, _, _, l2dtmdtn = Lambda2Stores(M, theta)
    # horz edge terms
    lhmn, _, _, lhdtmdtn = LambdaHStores(M, theta)
    # vert edge terms
    lvmn, _, _, lvdtmdtn = LambdaVStores(M, theta)
    
    # now, reapply pmpn to each term and construct the norms in the whole space
    # remember, we need to replace pmpn in our terms now
    # NOTE: If we aren't treating p_m=1 for each pm, you'll need to uncomment the multiplication by pmpn
    ip = ( l2mn + lhmn + lvmn ) #* pmpn
    ipDT = ( l2dtmdtn + lhdtmdtn + lvdtmdtn ) #* pmpn
    return ip, ipDT

#%% Minimisation problem functions; objective function, constraint builders

def JandJac(UU, ipDT):
    '''
    Evaluates the objective function J and it's Jacobian as a function of UU, returning them as a tuple.
    INPUTS:
        UU: (2M,) float, representing the coefficient vector. Should be the case that U = Real2Comp(UU).
        ipDT: (M^2-1,M^2-1) float, matrix where entry (m,n) is <\grad^\theta phi_m, \grad^\theta phi_n>_{\compMes}.
    OUTPUTS:
        Jval: float, the value of the objective function. By definition, this is real.
        JacVal: (2M,) float, value of the jacobian of the objective function, which is again real by definition
    '''
    
    U = Real2Comp(UU)
    # casting to real is safe, since this expression is always real by definition
    Jval = np.real( np.sum( np.outer(U, np.conjugate(U)) * ipDT ) )
    
    JacVal = np.zeros_like(UU, dtype=float)
    # shouldn't have to cast to real since we only deal with real and imag parts anyway
    JacVal[0::2] = 2. * ( np.matmul(np.real(ipDT),UU[0::2]) + np.matmul(np.imag(ipDT),UU[1::2]) )
    JacVal[1::2] = 2. * ( np.matmul(np.real(ipDT),UU[1::2]) - np.matmul(np.imag(ipDT),UU[0::2]) )
    
    return Jval, JacVal

def PeriodicConstraints(M):
    '''
    Given the highest order polynomial we are using to approximate, create the constraints corresponding to the
    forcing of periodicity of the solution.
    INPUTS:
        M: int, M-1 is the highest power in the polynomial approximation we are using.
    OUTPUTS:
        ConstraintMatrix: (4M,2M^2) int, the matrix defining the linear constraint.
    '''
    
    # we will have 4M conditions: there are 2 periodicity conditions giving us M equations each,
    # and then we have to split into real and imaginary parts
    # slice convention: start:stop:step, where we DONT INCLUDE stop!
    
    # on the plus side, our constraint matrix is a matrix of integers!
    # 4M constraints (rows) relating 2M^2 (size of UU) variables (columns)
    ConstraintMatrix = np.zeros((4*M, 2*M*M), dtype=int)

    # these are the periodicity constraints along the x-boundaries
    for jm in range(M):
        ##REAL PART
        # 1st one appears at index 2(jm + M)
        # go up in steps of size 2M
        # last one appears at index 2(jm + M(M-1))
        ConstraintMatrix[2*jm , 2*(jm+M):2*(jm+M*(M-1))+1:2*M] = np.ones((M-1), dtype=int)
        
        ##IMAG PART
        # 1st one appears at index 2(jm + M)+1
        # go up in steps of size 2M
        # last one appears at index 2(jm + M(M-1))+1      
        ConstraintMatrix[2*jm+1 , 2*(jm+M)+1:2*(jm+M*(M-1))+2:2*M] = np.ones((M-1), dtype=int)
    
    # now for the periodicity constraints along the y-boundaries
    for im in range(M):
        ##REAL PART
        # 1st one appears at index 2(im*M + 1)
        # go up in steps of 2
        # last one appears at index 2(im*M + M-1)
        ConstraintMatrix[2*M + 2*im , 2*(im*M + 1):2*(im*M + M-1)+1:2] = np.ones((M-1), dtype=int)
        
        ##IMAG PART
        # 1st one appears at index 2(im*M + 1)+1
        # go up in steps of 2
        # last one appears at index 2(im*M + M-1)+1
        ConstraintMatrix[2*M + 2*im + 1 , 2*(im*M + 1)+1:2*(im*M + M-1)+2:2] = np.ones((M-1), dtype=int)

    return ConstraintMatrix

def OrthogonalityConstraints(M, prevUs, ip):
    '''
    Given the highest order polynomial we are using to approximate, and the previous polynomials that
    we are required to be orthogonal to,
    create the constraints corresponding to the orthogonality of u to each of the u^l.
    INPUTS:
        M: int, M-1 is the highest power in the polynomial approximation we are using.
        prevUs: (l,2M^2) float, row p is the vector UUp for 1<=p<=l.
        ip: (M^2-1,M^2-1) float, matrix where entry (m,n) is <phi_m, phi_n>_{\compMes}.
    OUTPUTS:
        ConstraintMatrix: (2l,2M^2) float, the matrix defining the linear constraint.
    '''
    
    lMax = np.shape(prevUs)[0]
    ConstraintMatrix = np.zeros((2*lMax, 2*M*M), dtype=float)
    
    for l in range(lMax):
        # construct the rows that correspond to orthogonality to prevUs[l]
        UUl = prevUs[l,:]
        # this vector is such that pReal[m] = sum_n( UUl[2n]ip[m,n] )
        pReal = np.sum( UUl[0::2]*ip, axis=1 )
        # this vector is such that pImag[m] = sum_n( UUl[2n+1]ip[m,n] )
        pImag = np.sum( UUl[1::2]*ip, axis=1 )
        
        #REAL PART
        ConstraintMatrix[2*l,0::2] = pReal #UU[2m] has coefficient sum_n( UUl[2n]*ip[m,n] )
        ConstraintMatrix[2*l,1::2] = pImag #UU[2m+1] has coefficient sum_n( UUl[2n+1]*ip[m,n] )
        #IMAG PART
        ConstraintMatrix[2*l+1,0::2] = - pImag #UU[2m] has coefficient -sum_n( UUl[2n+1]ip[m,n] )
        ConstraintMatrix[2*l+1,1::2] = pReal ##UU[2m+1] has coefficient sum_n( UUl[2n]ip[m,n] )
        
    return ConstraintMatrix

def BuildLinearConstraint(M, ip, prevUs=0):
    '''
    Build the linear constraint matrix for the minimisation problem.
    The first 4M rows are the periodicity constraints, and will be build by PeriodicConstraints.
    The remaining rows are the orthogonality conditions given the previous functions in prevUs,
    there will be no additional rows is prevUs is not provided (assuming no orthogonality conditions to be met).
    INPUTS:
        M: int, M-1 is the highest power in the polynomial approximation we are using.
        prevUs: (l,2M^2) float, row p is the vector UUp for 1<=p<=l.
        ip: (M^2-1,M^2-1) float, matrix where entry (m,n) is <phi_m, phi_n>_{\compMes}.
    OUTPUTS:
        LinearConstraint: scipy.optimize.LinearConstraint, specifying the periodicity and orthogonality constraints.
        constraintMatrix: (4M+2l, 2*M*M) float, the matrix specifying the constraint, for error checking purposes.
    '''
    
    MperiodicPart = PeriodicConstraints(M)
    if np.ndim(prevUs)==0:
        # if prevUs is only a scalar, then we haven't provided it (or have provided it improperly), 
        # so assume no orthogonality constraints
        constraintMatrix = MperiodicPart
    else:
        # we have some orthogonality conditions
        MorthPart = OrthogonalityConstraints(M, prevUs, ip)
        constraintMatrix = np.vstack((MperiodicPart, MorthPart))
    
    # note: lb can be a scalar if the bounds are all the same!
    return LinearConstraint(constraintMatrix, 0, 0), constraintMatrix

def NormConstraint(M, ip, giveJac=True):
    '''
    Given the highest order polynomial we are using to approximate, create the constraints corresponding to the
    forcing of the norm of the solution to be one.
    INPUTS:
        M: int, M-1 is the highest power in the polynomial approximation we are using.
        ip: (M^2-1,M^2-1) float, matrix where entry (m,n) is < phi_m, phi_n>_{\compMes}.
    OUTPUTS:
        NonlinearConstraint: scipy.optimize.NoninearConstraint, defines the norm constraint.   
    '''
    
    # we must construct a function f(UU) such that
    # lb <= f(UU) <= ub
    # is the form for our nonlinear constraints
    def f(UU):
        '''
        Computes the vector of constraints at the given value of UU
        '''
        
        U = Real2Comp(UU)
        # cast to real should be safe, since this expression is always real by definition
        return np.real( np.sum( np.outer(U, np.conjugate(U)) * ip ) )
    
    # also provide the jacobian of this constraint
    def Jf(UU):
        '''
        Computes the Jacobian of the vector of constraints at the given value of UU
        '''
        
        JacVal = np.zeros_like(UU, dtype=float)
        # shouldn't have to cast to real since we only deal with real and imag parts anyway
        JacVal[0::2] = 2. * ( np.matmul(np.real(ip),UU[0::2]) + np.matmul(np.imag(ip),UU[1::2]) )
        JacVal[1::2] = 2. * ( np.matmul(np.real(ip),UU[1::2]) - np.matmul(np.imag(ip),UU[0::2]) )
        # return as a (2M,) vector
        return JacVal#.reshape((1,np.shape(UU)[0]))
    
    # the lower bound and the upper bound coincide at 1, so just provide scalars here
    if giveJac:
        return NonlinearConstraint(f, 1, 1, jac=Jf), f
    else:
        return NonlinearConstraint(f, 1, 1), f
    
#%% Poly2D class

class Poly2D:
    '''
    Stores 2D polynomials of the form
        u(x,y) = \sum_{m=0}^{M^2-1} u_m\phi_m,
    for the basis functions phi_m defined above.
    ATTRIBUTES:
        uCoeffs: (M*M,) complex, the coefficients in the representation
        M: int, M-1 is the highest-order term of the polynomial
        theta: (2,) float, the value of the quasi-momentum that this polynomial was computed at
        lbda: float, the "eigenvalue" (or value of the objective function) corresponding to this polynomial
    METHODS:
        val(x,y): evaluates the polynomial the the coordinate pairs (x,y)
    '''
      
    # initialisation method
    def __init__(self, theta, U):
        '''
        Initialisation method for an instance of the class.
        INPUTS:
            theta: (2,) float, the value of the quasi-momentum that this polynomial was computed at
            U: (M*M,) complex, the coefficients in the representation of the function
        OUTPUTS:
            Poly2D: instance of class, the values of M and lbda are set automatically
        '''
        
        # QM is given to us for free
        self.theta = theta
        # as is the vector of coefficients
        self.uCoeffs = U
        # can deduce M from the length of U
        self.M = int( np.sqrt(np.shape(U)[0]) )
        # finally, record the value of lbda
        self.lbda = self._J(U)
        # we are now all done
        return
    
    def __str__(self):
        '''
        Default print output when instance of class is passed to print()
        '''
        rFig, iFig = self.plot()
        rFig[0].show()
        iFig[0].show()
        return 'Poly2D with M = %d' % self.M
    
    def _J(self, U):
        '''
        Returns the value of lbda=omega^2 by evaluating the objective function.
        INPUTS:
            U: (2M,) float, representing the coefficient vector.
            ipDT: (M^2-1,M^2-1) float, matrix where entry (m,n) is 
                    <\grad^\theta phi_m, \grad^\theta phi_n>_{\compMes}.
        OUTPUTS:
            Jval: float, the value of the objective function. By definition, this is real, and equal to lbda.
        '''
        
        _, ipDT = TLambdaStores(self.M, self.theta)
        return np.real( np.sum( np.outer(U, np.conjugate(U)) * ipDT ) )
        
    def val(self, x, y):
        '''
        Evaluates the Poly2D at the point(s) (x,y)
        INPUTS:
            x,y: (l,) floats, (x,y) coordinate pairs as 1D arrays.
        OUTPUTS:
            polyVals: (l,) complex, values of the Poly2D at the input co-ordinates.
        '''
        
        polyVals = np.zeros_like(x, dtype=complex)
        for m in range(self.M**2):
            jm = m % self.M
            im = m // self.M
            polyVals += self.uCoeffs[m] * (x**im) * (y**jm)
        return polyVals

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

        x = y = np.linspace(0,1,num=N)
        X, Y = np.meshgrid(x,y)
        U = self.val(X,Y)
        rCon = rAx.contourf(X, Y, np.real(U), levels=levels)
        iCon = iAx.contourf(X, Y, np.imag(U), levels=levels)
        # make colourbars
        rF.colorbar(rCon)
        iF.colorbar(iCon)
        return [rF, rAx], [iF, iAx]
    
    def ip(self, p):
        '''
        Computes the inner product (L^2(compMes)) of this Poly2D with another Poly2D, p.
        Note that p must have the same values of M and theta to self.
        INPUTS:
            p: Poly2D, computes the inner product <self, p>_L^2(compMes)
        OUTPUTS:
            iVal: complex, the value of the inner product <self, p>_L^2(compMes)
        '''
        
        if (self.M!=p.M) or (not np.any(np.isclose(self.theta,p.theta))):
            raise ValueError('Cannot compute inner product for Poly2D with different M or theta values')
        ip, _ = TLambdaStores(self.M, self.theta)
        iVal = np.sum( np.outer(self.uCoeffs, np.conjugate(p.uCoeffs)) * ip )
        return iVal

#%% Solver for analysis export: allows us to solve the variational problem starting from a certain state
def VarProb(M, N, theta, nIts=2500, prevUs=[], x0=[], lOff=False):
    '''
    Solves the variational problem with the parameters provided.
    INPUTS:
        M: int, M-1 is the highest order term in the polynomial approximation
        N: int, number of eigenfunctions (and values) to find
        theta: (2,) float, value of the quasi-momentum
        nIts: int, maximum number of iterations for minimise solve
        prevUs: list of Poly2D, storing the eigenfunctions that we previously found. If this is passed, we will find the next N eigenvalues, rather than starting from the 0th eigenvalue
        x0: (M*M,) complex, starting guess for convergence scheme. If blank, we use the constant function as the starting guess
        log: bool, if True then minimiser will NOT write progress to screen
    OUTPUTS:
        uRealStore: (N,2M^2) float, the computed solutions to the minimisation problem stacked row-wise
        omegaSqStore: (N,) float, the values of omega^2, or the objective function at the solution
        noConv: int, the run failed to converge for n=noConv (which will terminate the loop early). If noConv=-1, then all eigenfunctions were found sucessfully
    '''    
    
    # create solver options handle
    options = {'maxiter' : nIts, 'disp' : (not lOff) }
    
    # create flag for no convergence: assume everything went well
    noConv = -1
    
    # deterine additional information that has been passed in
    nPrev = len(prevUs) # we have been given this many previous eigenfunctions
    ## Initialise storage for the solutions
    # (l,2m) + (l,2m+1) is the coefficient u^l_m
    if nPrev==0:
        # if we were given no previous eigenfunctions, setup storage as usual
        uRealStore = np.zeros((N, 2*M*M), dtype=float)
    else:
        # if we were given some previous eigenfunctions, setup storage differently
        uRealStore = np.zeros((N+nPrev, 2*M*M), dtype=float)
        # store the previous eigenfunctions in our storage matrix
        for i,p in enumerate(prevUs):
            uRealStore[i,:] = Comp2Real(p.uCoeffs)
    # we are always trying to find N eigenvalues
    omegaSqStore = np.zeros((N,), dtype=float)
    
    # Compute inner products
    ip, ipDT = TLambdaStores(M, theta)
    # Set objective function
    JandJacOpt = lambda UU: JandJac(UU, ipDT)
    # If provided, use the given initial guess, otherwise use the constant function
    if np.any(x0):
        UU0 = Comp2Real(x0)
    else:    
        # Set initial guess to be the constant function
        U0 = np.zeros((M*M,), dtype=complex)
        U0[0] += 1./np.sqrt(3.)
        UU0 = Comp2Real(U0)
    
    ## Solving begins

    # We can setup the norm constraint independently
    nonLinCon, _ = NormConstraint(M, ip)
    
    if not lOff:
        print('Beginning solve [n]...')
    
    # find the first eigenfunction, either u^0 or u^nPrev depending on our inputs, outside the loop
        print(' ----- \n %d + [0] ' % nPrev)
    if nPrev==0:
        # no previous eigenvalues to pass in 
        linCon, _ = BuildLinearConstraint(M, ip)
    else:
        # pass in the eigenvalues that we already were given
        linCon, _ = BuildLinearConstraint(M, ip, prevUs=uRealStore[:nPrev,:])
    t0 = time.time()
    resultJ = minimize(JandJacOpt, UU0, constraints=[linCon, nonLinCon], jac=True, options=options, method='SLSQP')
    t1 = time.time()
    # save the coefficients for use in the next solve, note we save these to row nPrev regardless of whether or not we had any previous eigenfunctions passed in
    uRealStore[nPrev,:] = resultJ.x
    # save the eigenvalue
    omegaSqStore[0] = resultJ.fun
    if not lOff:
        print('Runtime: approx %d mins (%s seconds) \n -----' % (np.round((t1-t0)/60), t1-t0))
    if resultJ.status!=0:
        # didn't converge, flag this
        noConv = 0
        print('Failed to converge when finding e-function n=%d+0' % nPrev)
    else:
        # for every eigenfunction and value that we want above the first
        for n in range(1,N):
            if not lOff:
                print(' ----- \n [%d]' % n)
            # setup and solve minimisation problem, we are now constrained by the eigenfunctions previously passed in and the ones we have computed
            linCon, _ = BuildLinearConstraint(M, ip, prevUs=uRealStore[:nPrev+n,:])
            t0 = time.time()
            resultJ = minimize(JandJacOpt, UU0, constraints=[linCon, nonLinCon], jac=True, options=options, method='SLSQP')
            t1 = time.time()
            # save coefficients
            uRealStore[nPrev+n,:] = resultJ.x
            # save eigenvalue
            omegaSqStore[n] = resultJ.fun
            if not lOff:
                print('Runtime: approx %d mins (%s seconds) \n -----' % (np.round((t1-t0)/60), t1-t0))
            if resultJ.status!=0:
                # didn't converge, flag and break loop
                noConv = n
                print('Failed to converge when finding e-function n=%d+%d' % (nPrev,n))
                break
    
    # do not return the eigenfunctions that were already passed in as known
    if not lOff:
        print('NOTE: noConv is relative to 1st eigenvalue computed, GIVEN previous eigenfunctions that were passsed.')
    return uRealStore[nPrev:,:], omegaSqStore, noConv

#%% Wrappers for command-line execution, so that imports can also be used

# Solve via local minimisation: faster, but will become inaccurate a lot quicker
def SolveVarProb(M, N, theta, nIts=2500, lOff=False):
    '''
    DEPRICIATED - USE VarProb or GlobalVarProbSolve as alternatives
    Solves the variational problem with the parameters provided.
    Wrapper function for command-line script
    INPUTS:
        M: int, M-1 is the highest order term in the polynomial approximation
        N: int, number of eigenfunctions (and values) to find
        theta: (2,) float, value of the quasi-momentum
        nIts: int, maximum number of iterations for minimise solve
        log: bool, if True then minimiser will NOT write progress to screen
    OUTPUTS:
        uRealStore: (N,2M^2) float, the solutions to the minimisation problem stacked row-wise
        omegaSqStore: (N,) float, the values of omega^2, or the objective function at the solution
        noConv: int, the run failed to converge for n=noConv (which will terminate the loop early). If noConv=-1, then all eigenfunctions were found sucessfully
    '''
    
    # create solver options handle
    options = {'maxiter' : nIts, 'disp' : (not lOff) }
    
    # create flag for no convergence: assume everything went well
    noConv = -1
    
    ## Initialise storage for the solutions
    # (l,2m) + (l,2m+1) is the coefficient u^l_m
    uRealStore = np.zeros((N, 2*M*M), dtype=float)
    omegaSqStore = np.zeros((N,), dtype=float)
    
    # Compute inner products
    ip, ipDT = TLambdaStores(M, theta)
    # Set objective function
    JandJacOpt = lambda UU: JandJac(UU, ipDT)
    # Set initial guess to be the constant function
    U0 = np.zeros((M*M,), dtype=complex)
    U0[0] += 1./np.sqrt(3.)
    UU0 = Comp2Real(U0)
    
    ## Solving begins
    
    # solve each minimisation problem in sequence,
    # starting with finding u^1, which adheres to no orthogonality conditions,
    # up to finding u^n, which adheres to N-1 orthogonality conditions
    
    # We can setup the norm constraint independently though
    nonLinCon, _ = NormConstraint(M, ip)
    
    print('Beginning solve [n]...')
    
    # for ease, we find u^1 outside the loop
    print(' ----- \n [0] ')
    linCon, _ = BuildLinearConstraint(M,ip)
    t0 = time.time()
    resultJ = minimize(JandJacOpt, UU0, constraints=[linCon, nonLinCon], jac=True, options=options, method='SLSQP')
    t1 = time.time()
    # save the coefficients for use in the next solve
    uRealStore[0,:] = resultJ.x
    # save the eigenvalue
    omegaSqStore[0] = resultJ.fun
    print('Runtime: approx %d mins (%s seconds) \n -----' % (np.round((t1-t0)/60), t1-t0))
    if resultJ.status!=0:
        # didn't converge, flag this
        noConv = 0
        print('Failed to converge when finding e-function n=0')
    else:
        # for every eigenfunction and value that we want above the first
        for n in range(1,N):
            print(' ----- \n [%d]' % n)
            # setup and solve minimisation problem with n constraints
            linCon, _ = BuildLinearConstraint(M, ip, prevUs=uRealStore[:n,:])
            t0 = time.time()
            resultJ = minimize(JandJacOpt, UU0, constraints=[linCon, nonLinCon], jac=True, options=options, method='SLSQP')
            t1 = time.time()
            # save coefficients
            uRealStore[n,:] = resultJ.x
            # save eigenvalue
            omegaSqStore[n] = resultJ.fun
            print('Runtime: approx %d mins (%s seconds) \n -----' % (np.round((t1-t0)/60), t1-t0))
            if resultJ.status!=0:
                # didn't converge, flag and break loop
                noConv = n
                print('Failed to converge when finding e-function n=%d' % n)
                break
    
    return uRealStore, omegaSqStore, noConv

# Global minimisation via BasinHopping: slower, but will give us a better answer
def GlobalVarProbSolve(M, N, theta, nIts=10, nIts_inner=2500, prevUs=[], x0=[], lOff=False):
    '''
    Solves the variational problem with the parameters provided, using BasinHopping method for global minimisation.
    Can pass in additional previous eigenfunctions to start solving from "higher up" eigenfunctions.
    Note that this will be more accurate than just using SciPy's minimise function, however will take considerably longer!
    Serves as a wrapper function for command-line script.
    INPUTS:
        M: int, M-1 is the highest order term in the polynomial approximation
        N: int, number of eigenfunctions (and values) to find
        theta: (2,) float, value of the quasi-momentum
        nIts: int, maximum number of iterations for BasinHopping solve
        nIts_inner: int, maximum number of iterations for minimise solve
        prevUs: list of Poly2D, storing the eigenfunctions that we previously found. If this is passed, we will find the next N eigenvalues, rather than starting from the 0th eigenvalue
        x0: (M*M,) complex, starting guess for convergence scheme. If blank, we use the constant function as the starting guess
        lOff: bool, if True then minimiser will NOT write progress to screen
    OUTPUTS:
        uRealStore: (N,2M^2) float, the solutions to the minimisation problem stacked row-wise
        omegaSqStore: (N,) float, the values of omega^2, or the objective function at the solution
        noConv: int, the run failed to converge for n=noConv (which will terminate the loop early). If noConv=-1, then all eigenfunctions were found sucessfully
    '''    
    
    # create solver options handle
    options = {'maxiter' : nIts_inner, 'disp' : False }
    
    # create flag for no convergence: assume everything went well
    noConv = -1
    
    # deterine additional information that has been passed in
    nPrev = len(prevUs) # we have been given this many previous eigenfunctions
    ## Initialise storage for the solutions
    # (l,2m) + (l,2m+1) is the coefficient u^l_m
    if nPrev==0:
        # if we were given no previous eigenfunctions, setup storage as usual
        uRealStore = np.zeros((N, 2*M*M), dtype=float)
    else:
        # if we were given some previous eigenfunctions, setup storage differently
        uRealStore = np.zeros((N+nPrev, 2*M*M), dtype=float)
        # store the previous eigenfunctions in our storage matrix
        for i,p in enumerate(prevUs):
            uRealStore[i,:] = Comp2Real(p.uCoeffs)
    # we are always trying to find N eigenvalues
    omegaSqStore = np.zeros((N,), dtype=float)
    
    # Compute inner products
    ip, ipDT = TLambdaStores(M, theta)
    # Set objective function
    JandJacOpt = lambda UU: JandJac(UU, ipDT)
    # If provided, use the given initial guess, otherwise use the constant function
    if np.any(x0):
        UU0 = Comp2Real(x0)
    else:    
        # Set initial guess to be the constant function
        U0 = np.zeros((M*M,), dtype=complex)
        U0[0] += 1./np.sqrt(3.)
        UU0 = Comp2Real(U0)
    
    # setup kwargs dictionary to be passed into minimize on iterations
    # NOTE: we'll need to redefine the constraints value each time we move onto a new eVal
    minKWs = {}
    minKWs['jac'] = True
    minKWs['options'] = options
    minKWs['method'] = 'SLSQP'    
    ## Solving begins
    
    # solve each minimisation problem in sequence,
    # starting with finding u^1, which adheres to no orthogonality conditions,
    # up to finding u^n, which adheres to N-1 orthogonality conditions
    
    # We can setup the norm constraint independently though
    nonLinCon, _ = NormConstraint(M, ip)
    
    print('Beginning solve %d + [n]...' % nPrev)
    
    # find the first eigenfunction, either u^0 or u^nPrev depending on our inputs, outside the loop
    if nPrev==0:
        # no previous eigenvalues to pass in 
        linCon, _ = BuildLinearConstraint(M, ip)
    else:
        # pass in the eigenvalues that we already were given
        linCon, _ = BuildLinearConstraint(M, ip, prevUs=uRealStore[:nPrev,:])
    print(' ----- \n %d + [0] ' % nPrev)
    minKWs['constraints'] = [linCon, nonLinCon]
    t0 = time.time()
    resultJ = basinhopping(JandJacOpt, UU0, niter=nIts, minimizer_kwargs=minKWs, niter_success=5, disp=not lOff)
    t1 = time.time()
    # save the coefficients for use in the next solve
    uRealStore[nPrev,:] = resultJ.x
    # save the eigenvalue
    omegaSqStore[0] = resultJ.fun
    print('Runtime: approx %d mins (%s seconds) \n -----' % (np.round((t1-t0)/60), t1-t0))
    if resultJ.lowest_optimization_result.status!=0:
        # didn't converge, flag this
        noConv = 0
        print('Failed to converge when finding e-function n=%d+0' % nPrev)
    else:
        # for every eigenfunction and value that we want above the first
        for n in range(1,N):
            print(' ----- \n %d + [%d]' % (nPrev,n))
            # setup and solve minimisation problem with nPrev+n constraints
            linCon, _ = BuildLinearConstraint(M, ip, prevUs=uRealStore[:nPrev+n,:])
            minKWs['constraints'] = [linCon, nonLinCon]
            t0 = time.time()
            resultJ = basinhopping(JandJacOpt, UU0, niter=nIts, minimizer_kwargs=minKWs, niter_success=5, disp=not lOff)
            t1 = time.time()
            # save coefficients
            uRealStore[nPrev+n,:] = resultJ.x
            # save eigenvalue
            omegaSqStore[n] = resultJ.fun
            print('Runtime: approx %d mins (%s seconds) \n -----' % (np.round((t1-t0)/60), t1-t0))
            if resultJ.lowest_optimization_result.status!=0:
                # didn't converge, flag and break loop
                noConv = n
                print('Failed to converge when finding e-function n=%d+%d' % (nPrev,n))
                break
    
    return uRealStore[nPrev:,:], omegaSqStore, noConv

    
#%% Command-line execution

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Approximation of eigenvalues and eigenfunctions for the Dirichlet-to-Neumann map, for the Cross-in-plane geometry.')
    parser.add_argument('-M', type=int, help='<Required> M-1 is the highest order term of the polynomial approximation.', required=True)
    parser.add_argument('-N', type=int, help='<Required> Number of eigenfunctions and eigenvalues to find (starting from lowest eigenvalue)', required=True)
    parser.add_argument('-fn', default='', type=str, help='Output file name, if blank filename will be auto-generated')
    parser.add_argument('-fd', default='./CompMesVarProb_Results/', type=str, help='Directory to save output file to.')
    parser.add_argument('-t1', default=0.0, type=float, help='QM_1 will be set to this value multiplied by pi.')
    parser.add_argument('-t2', default=0.0, type=float, help='QM_2 will be set to this value multiplied by pi.')
    parser.add_argument('-local', action='store_true', help='If passed, we will only solve for a single local solution, rather than attempting to find a global minimiser.')
    parser.add_argument('-nIts', default=10, type=int, help='Maximum number of iterations for BasinHopping')
    parser.add_argument('-nIts_inner', default=2500, type=int, help='Maximum number of iterations for minimise')
    parser.add_argument('-lOff', action='store_true', help='Suppress printing of progress and solver log to the screen')

    # extract input arguments and get the setup ready
    args = parser.parse_args()

    # There will be (2M+1)^2 Fourier modes, the "highest" being of order M
    M = args.M
    # We want to find this many eigenfunctions
    N = args.N
    # There are a total of 2M (periodicity) + N-1 (orthogonality) + 1 (norm) constraints,
    # and we have a space of dimension M^2.
    # We should check if we can actually find a solution to this problem
    if M**2 < 2*M + N:
        raise ValueError('M^2 (%d) < (%d) 2M + N: cannot find orthogonal functions. ABORT' % (M**2, N))
    else:
        print('Will find %d eigenpairs, approximating with %d-dimensional Polynomial space' % (N, M**2))

    # quasi-momentum value
    theta = np.array([args.t1, args.t2], dtype=float) * pi
    print('Read theta = [%.3f, %.3f]\pi' % (args.t1, args.t2))

    if args.local:
        # local solve requested, not recommended though
        print('Local solve requested')
        uRealStore, _, noConv = SolveVarProb(M, N, theta, nIts=args.nIts_inner, lOff=args.lOff)
    else:
        uRealStore, _, noConv = GlobalVarProbSolve(M, N, theta, nIts=args.nIts, nIts_inner=args.nIts_inner, lOff=args.lOff)
    
    # run is now complete - save the output if we converged!
    if (not noConv):
        infoFile = SaveRun(theta, uRealStore, fname=args.fn, fileDump=args.fd)
        # inform the user of completion of the script
        print('Completed, output file saved to:' + infoFile)
        with open('CompMesVarProb-ConveyerFile.txt', 'w') as fh:
            fh.write("%s" % infoFile)
            sys.exit(0)
    else:
        sys.exit(1)