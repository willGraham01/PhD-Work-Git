#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 09:47:34 2021

@author: will

This script will attempt to solve the system of equations on the composite medium (dielectric plus dielectic inclusions) for the cross-in-plane geometry example.
The domain is $\Omega = \left[0,1\right]^2$ with periodic boundary conditions.
The quasi-momentum $\theta$ is fixed for each problem.

See the notebook CompositeMedium_PeriodicFDM.ipynb for full documentation.
This script is designed to be run from the command line, and to be passed the value of N to use.
The quasi-momentum and value of alpha3 have to be manually set within the script.
"""

# imports for command line runs
#import sys
# sys.argv[1:] contains command line arguments. Index 0 is just the script name
import argparse

# imports for construction and analysis
import numpy as np
from numpy.random import choice, uniform
from scipy.linalg import eig

import matplotlib.pyplot as plt
from matplotlib import rc, cm
rc('text', usetex=True)
from mpl_toolkits.mplot3d import Axes3D

# for saving data
from datetime import datetime

# CONSTRUCTION OF FDM functions
def RegionalFDM(M, region, N, theta, log=False):
    '''
    Construct the FDM rows that correspond to gridpoints i,j in the region Omega_region.
    The input matrix M will be overwritten.
    INPUTS:
        M: (N*N,N*N) complex, the FDM into which the values for the rows will be written
        region: int, 1-4 specifying which region to construct the rows for
        N: int, number of meshpoints in each dimension
        theta: (2,) float, the value of the quasi-momentum
    OUTPUTS:
        M: Rows associated with gridpoints in the region Omega_region will be populated with FDM entries. 
        Overwrites any values previously stored in entries that are to be edited.
    '''
    
    # Compute some common constants first
    h = 1./(N-1)
    sqMagTheta = theta[0]*theta[0] + theta[1]*theta[1]
    
    # determine correct indices to loop over via lookup
    if region==1:
        jR = range(N//2)
        iR = range(N//2)
    elif region==2:
        jR = range(1+N//2,N-1)
        iR = range(N//2)
    elif region==3:
        jR = range(1+N//2,N-1)
        iR = range(1+N//2,N-1)
    elif region==4:
        jR = range(N//2)
        iR = range(1+N//2,N-1)
    else:
        raise ValueError('Unrecognised region, got %d' % (region))
        
    # now insert the entries of the FDM for the given region
    # The i + N*j th row of FDM contains the equation for the gridpoint u_{i,j}.
    for j in jR:
        for i in iR:
            # this is the row that we are going to set
            Mrow = M2C(i,j)
            # u_{i,j} prefactor is placed on the diagonal
            M[Mrow, Mrow] = sqMagTheta + 4/(h*h)
            # now place u_{i-1,j} and u_{i+1,j} prefactors
            # take index modulo N, since if you're on the periodic boundary you need to 
            # loop to the other side of the domain
            iLeft = (i-1) % (N-1)
            iRight = (i+1) % (N-1)
            leftIndex = M2C(iLeft, j)
            rightIndex = M2C(iRight, j)
            M[Mrow, leftIndex] = - ( 1./h - 1.j*theta[0] ) / h
            M[Mrow, rightIndex] = - ( 1./h + 1.j*theta[0] ) / h
            # now place u_{i,j-1} and u_{i,j+1} prefactors, again accounting for periodic boundary
            jUp = (j+1) % (N-1)
            jDown = (j-1) % (N-1)
            upIndex = M2C(i,jUp)
            downIndex = M2C(i,jDown)
            M[Mrow, upIndex] = - ( 1./h + 1.j*theta[1] ) / h
            M[Mrow, downIndex] = - ( 1./h - 1.j*theta[1] ) / h
            # this completes the construction of this row of the FDM, we now have that
            # FDM[Mrow,:] * U[Mrow] defines the expression for -\Laplacian_{\theta} at i,j.
            if log:
                print('---')
                print('Examined i,j = (%d,%d), Mrow = %d' % (i,j, Mrow))
                print('left: %d, right: %d, up: %d, down: %d' % (leftIndex, rightIndex, upIndex, downIndex))
                print('set values to l/r/u/p/HERE : ', M[Mrow, leftIndex], M[Mrow, rightIndex], M[Mrow, upIndex], 
                 M[Mrow, downIndex], M[Mrow, Mrow])
    if log: 
        print('Finished region: %d' % (region))
    return

def HorEdgeFDM(M, edge, N, theta, log=False):
    '''
    Construct the FDM rows that correspond to gridpoints i,j on the edge I_{edge}
    The input matrix M will be overwritten.
    INPUTS:
        M: (N*N,N*N) complex, the FDM into which the values for the rows will be written
        edge: int, either 23 or 34, specifying which horizontal edge to construct the rows for
        N: int, number of meshpoints in each dimension
        theta: (2,) float, the value of the quasi-momentum
    OUTPUTS:
        M: Rows associated with gridpoints in the edge I_{edge} will be populated with FDM entries. 
        Overwrites any values previously stored in entries that are to be edited.
    '''
    
    h = 1./(N-1)
    # setup loop ranges based on the edge we are working on
    if edge==23:
        iR = range(N//2)
    elif edge==34:
        iR = range(1+N//2,N-1)
    else:
        raise ValueError('Unknown horizontal edge, got %d' % (edge))
    j = N//2 # j is constant for horizontal edges
    
    # now insert the entries of the FDM for the given edge
    # The i + N*j th row of FDM contains the equation for the gridpoint u_{i,j}.
    for i in iR:
        # get row index
        Mrow = M2C(i,j)
        # u_{i,j} prefactor is placed on the diagonal
        M[Mrow, Mrow] = theta[0]*theta[0] + 2./h + 2./(h*h)
        # now place u_{i-1,j} and u_{i+1,j} prefactors
        # take index modulo N, since if you're on the periodic boundary you need to 
        # loop to the other side of the domain
        iLeft = (i-1) % (N-1)
        iRight = (i+1) % (N-1)
        leftIndex = M2C(iLeft, j)
        rightIndex = M2C(iRight, j)
        M[Mrow, leftIndex] = - ( 1./h - 1.j*theta[0] ) / h
        M[Mrow, rightIndex] = - ( 1./h + 1.j*theta[0] ) / h
        # now place u_{i,j-1} and u_{i,j+1} prefactors, again accounting for periodic boundary
        jUp = (j+1) % (N-1)
        jDown = (j-1) % (N-1)
        upIndex = M2C(i, jUp)
        downIndex = M2C(i, jDown)
        M[Mrow, upIndex] = - 1./h
        M[Mrow, downIndex] = - 1./h
        # this completes the construction of this row of the FDM, we now have that
        # FDM[Mrow,:] * U[Mrow] defines the expression for the edge operator at i,j.
        if log:
            print('---')
            print('Examined i,j = (%d,%d), Mrow = %d' % (i,j, Mrow))
            print('left: %d, right: %d, up: %d, down: %d' % (leftIndex, rightIndex, upIndex, downIndex))
            print('set values to l/r/u/p/HERE : ', M[Mrow, leftIndex], M[Mrow, rightIndex], M[Mrow, upIndex], 
                 M[Mrow, downIndex], M[Mrow, Mrow])
    if log:
        print('Finished edge: I_%d' % (edge))      
    return

def VerEdgeFDM(M, edge, N, theta, log=False):
    '''
    Construct the FDM rows that correspond to gridpoints i,j on the edge I_{edge}
    The input matrix M will be overwritten.
    INPUTS:
        M: (N*N,N*N) complex, the FDM into which the values for the rows will be written
        edge: int, either 31 or 53, specifying which vertical edge to construct the rows for
        N: int, number of meshpoints in each dimension
        theta: (2,) float, the value of the quasi-momentum
    OUTPUTS:
        M: Rows associated with gridpoints in the edge I_{edge} will be populated with FDM entries. 
        Overwrites any values previously stored in entries that are to be edited.
    '''
    
    h = 1./(N-1)
    # setup loop ranges based on the edge we are working on
    if edge==53:
        jR = range(N//2)
    elif edge==31:
        jR = range(1+N//2,N-1)
    else:
        raise ValueError('Unknown vertical edge, got %d' % (edge))
    i = N//2 # i is constant for vertical edges
    
    # now insert the entries of the FDM for the given edge
    # The i + N*j th row of FDM contains the equation for the gridpoint u_{i,j}.
    for j in jR:
        # get row index
        Mrow = M2C(i,j)
        # u_{i,j} prefactor is placed on the diagonal
        M[Mrow, Mrow] = theta[1]*theta[1] + 2./h + 2./(h*h)
        # now place u_{i-1,j} and u_{i+1,j} prefactors
        # take index modulo N, since if you're on the periodic boundary you need to 
        # loop to the other side of the domain
        iLeft = (i-1) % (N-1)
        iRight = (i+1) % (N-1)
        leftIndex = M2C(iLeft, j)
        rightIndex = M2C(iRight, j)
        M[Mrow, leftIndex] = - 1./h
        M[Mrow, rightIndex] = - 1./h
        # now place u_{i,j-1} and u_{i,j+1} prefactors, again accounting for periodic boundary
        jUp = (j+1) % (N-1)
        jDown = (j-1) % (N-1)
        upIndex = M2C(i, jUp)
        downIndex = M2C(i, jDown)
        M[Mrow, upIndex] = - ( 1./h + 1.j*theta[1] ) / h
        M[Mrow, downIndex] = - ( 1./h - 1.j*theta[1] ) / h
        # this completes the construction of this row of the FDM, we now have that
        # FDM[Mrow,:] * U[Mrow] defines the expression for the edge operator at i,j.
        if log:
            print('---')
            print('Examined i,j = (%d,%d), Mrow = %d' % (i,j, Mrow))
            print('left: %d, right: %d, up: %d, down: %d' % (leftIndex, rightIndex, upIndex, downIndex))
            print('set values to l/r/u/p/HERE : ', M[Mrow, leftIndex], M[Mrow, rightIndex], M[Mrow, upIndex], 
                 M[Mrow, downIndex], M[Mrow, Mrow])
    if log:
        print('Finished edge: I_%d' % (edge))
    return

def v3FDM(M, a3, N, tol=1e-8, log=True):
    '''
    Construct the FDM row that corresponds to the gridpoint at the vertex v_3.
    Different behaviour occurs for when alpha_3 = 0 and when it is non-zero.
    The input matrix M will be overwritten.
    INPUTS:
        M: (N*N,N*N) complex, the FDM into which the values for the rows will be written.
        a3: float, the value of the coupling constant alpha_3 at v_3.
        N: int, number of meshpoints in each dimension.
    OUTPUTS:
        M: Row N//2+N*N//2 will be populated with FDM entries.
        Overwrites any values previously stored in entries that are to be edited.    
        
    If a3 = 0, then the row is assembled as discretised above.
    If a3 != 0, then the row is assembled with a 1/a3 prefactor in front of the discretisation above.
    '''
    
    h = 1./(N-1)
    # v_3 always lies at gridpoint (i,j) = (N//2, N//2)
    i = j = N//2
    Mrow = M2C(i,j)
    
    # determine indices that will be used for neighbouring points
    iLeft = (i-1) % (N-1)
    iRight = (i+1) % (N-1)
    jUp = (j+1) % (N-1)
    jDown = (j-1) % (N-1)
    leftIndex = M2C(iLeft, j)
    rightIndex = M2C(iRight, j)
    upIndex = M2C(i, jUp)
    downIndex = M2C(i, jDown)
    
    M[Mrow, Mrow] = 4./h
    M[Mrow, leftIndex] = - 1./h
    M[Mrow, rightIndex] = - 1./h
    M[Mrow, upIndex] = - 1./h
    M[Mrow, downIndex] = - 1./h
    
    if log:
        print('---')
        print('[v3] Examined i,j = (%d,%d), Mrow = %d' % (i,j, Mrow))
        print('left: %d, right: %d, up: %d, down: %d' % (leftIndex, rightIndex, upIndex, downIndex))
        print('set values to l/r/u/p/HERE : ', M[Mrow, leftIndex], M[Mrow, rightIndex], M[Mrow, upIndex], 
                 M[Mrow, downIndex], M[Mrow, Mrow]) 
    
    if np.abs(a3)<=tol:
        # alpha_3 is 0, notify user and construct LHS without alpha_3^-1 factor
        print('alpha_3 is zero: take care when solving discretised system.')
    else:
        print('alpha_3 non zero: assembling with prefactor 1/alpha_3 at v_3.')
        # rescale row by alpha_3
        M[Mrow,:] /= a3
    return

# ANALYSIS/CHECKING functions
def B(N):
    '''
    Constructs the matrix B defined above.
    INPUTS:
        N: int, number of meshpoints in each dimension.
    OUTPUTS:
        B: (N*N,N*N) complex, defined above.
    '''
    
    B = np.eye((N-1)*(N-1), dtype=complex)
    row = M2C(N//2, N//2)
    B[row, row] = 0. + 0.j
    return B

def Beta(z, N):
    '''
    Evaluates the matrix \beta defined above.
    Note that the argument is just z, so use the convention z = omega^2 here.
    INPUTS:
        z: complex, z=omega^2.
        N: int, number of meshpoints in each dimension.
    OUTPUTS:
        beta: (N*N,N*N) complex, defined above and evaluated at z.
    '''
    
    return z * B(N)

# This function is mainly a reality check to ensure that InsertSlaveNodes does it's job correctly.
def CheckPeriodicBCs(N, U, tol=1e-8):
    '''
    Given a solution vector for the Finite Difference approximation, check whether or not the
    solution satisfies the periodic boundary conditions at x=0 <-> x=N and y=0 <-> y=N.
    INPUTS:
        N: int, number of meshpoints in each dimension
        U: (N*N,) complex, solution vector to FDM approximation. Can also be of shape (N,N).
    OUTPUTS:
        tf: (2,) bool, True/False values depending on whether the periodic BCs are satisfied
    '''
    
    tf = np.array([True, True], dtype=bool)
    if U.ndim==1:
        # have been passed a column vector, need to reshape
        u = U.reshape((N,N))
    else:
        # should only have been passed a (N,N) array, so just copy this and use the values
        u = np.copy(U)
    uTB = u[N-1,:] - u[0,:]
    uLR = u[:,0] - u[:,N-1]
    tbMax = np.max(np.abs(uTB))
    lrMax = np.max(np.abs(uLR))
    
    if lrMax > tol:
        tf[0] = False
    if tbMax > tol:
        tf[1] = False
    
    return tf

# Function extracts only the real eigenvalues we get from the FDM
def RealEvalIndices(wVals, tol=1e-8):
    '''
    Given the eigenvalues that our FDM approximation believes solve the discretised problem, extract only those
    which are entirely real (to a given tolerance).
    Return a boolian array tf such that wVals[tf] returns the array slice of only the real eigenvalues
    INPUTS:
        wVals: (n,) complex, eigenvalues from which to find entirely real values
        tol: (optional) float - default 1e-8, tolerance within which to accept imaginary part = 0
    OUTPUTS:
        tf: (n,) bool, wVals[tf] returns the array slice of only the real eigenvalues in wVals
    '''
    
    return np.abs(np.imag(wVals))<=tol

# Filters out NaN and inf valued eigenvalues, because machines silly
def Purge(vals):
    '''
    Given an array of complex values, remove any infs or NaN values in the array.
    INPUTS:
        vals: (n,) complex, array of values possibly containing infs or NaNs
    OUTPUTS:
        v: (m,) complex, array of values in v that are not inf or NaN
        tf: (n,) bool, vals[tf] = v
    '''
    
    tf = ((~ np.isnan(vals)) & (~ np.isinf(vals)))
    v = vals[tf]
    return v, tf

def Herm(A):
    '''
    Returns Hermitian matrix (conjugate transpose) of the matrix A.
    INPUTS:
        A: (n,n) complex, complex matrix to transpose and conjugate
    OUTPUTS:
        AH: (n,n) complex, conjugate transpose of A
    '''
    return A.conj().T

# Checks whether the FDM is Hermitian or not - from our analytic expressions, I think it should be!
def IsHermitian(A, rtol=1e-5, atol=1e-8):
    '''
    Checks whether the matrix A is Hermitian to within the specified tolerances, 
    by comparing it elementwise to it's conjugate transpose.
    INPUTS:
        A: (n,n) complex, complex matrix to check is Hermitian
        rtol: float, relative tolerance
        atol: float, absolute tolerance
    For more information on rtol and atol, see numpy.allclose
    OUTPUTS:
        tf: bool, if True then A is Hermitian, otherwise False is returned.
    '''
    return np.allclose(A, Herm(A), rtol=rtol, atol=atol)

def IsSymmetric(A, rtol=1e-5, atol=1e-8):
    '''
    Checks whether the matrix A is symmetric to within the specified tolerances, 
    by comparing it elementwise to it's transpose.
    INPUTS:
        A: (n,n) complex, complex matrix to check is symmetric
        rtol: float, relative tolerance
        atol: float, absolute tolerance
    For more information on rtol and atol, see numpy.allclose
    OUTPUTS:
        tf: bool, if True then A is symmetric, otherwise False is returned.
    '''
    return np.allclose(A, A.T, rtol=rtol, atol=atol)

# PLOTTING AND VISUALISATION functions, although I doubt these will see use
def InsertSlaveNodes(N, U, mat=True):
    '''
    Given a (N-1)^2 solution vector U, insert the values at the periodic "slave" meshpoints to form a N*N matrix, 
    or (N*N) column vector
    INPUTS:
        N: int, number of meshpoints in each dimension
        U: ((N-1)**2,) complex, column vector of eigenfunction values at (non-slave) meshpoints
        mat: bool - default True, if True, return an (N,N) matrix and otherwise return a (N*N,) column vector
    OUTPUTS:
        u: (N,N) or (N*N,) complex, representation of the solution U with "slave" gridpoint values appended
    '''

    # setup solution
    u = np.zeros((N,N), dtype=complex)
    # setup view of U to make life easier
    Uview = U.reshape((N-1,N-1))
    # copy values across, don't edit column vector U
    u[:N-1,:N-1] = Uview
    # insert missing "right boundary"
    u[N-1,:N-1] = np.copy(Uview[0,:])
    # insert missing "top boundary"
    u[:N-1,N-1] = np.copy(Uview[:,0])
    # insert point at (N-1,N-1) which has not yet been set
    u[N-1,N-1] = np.copy(u[0,0])
    if (not mat):
        u.resize((N*N,))
    
    return u

def PlotFn(N, U, levels=10):
    '''
    Create contour plots for the eigenfunction U that is passed in.
    INPUTS:
        N: int, number of meshpoints in each dimension
        U: ((N-1)**2,) complex, column vector of eigenfunction values at (non-slave) meshpoints
        levels: (optional) int - default 10, number of contour levels to pass to contourf.
    OUTPUTS:
        Two lists of the form [fig, ax] for the handles of the matplotlib figures,
        the first containing the plot of the real part of U and 
        the second containing the plot of the imaginary part of U.
    '''
    
    # gridpoints used
    x = y = np.linspace(0,1, num=N)
    # restore "slave" meshpoints
    u = InsertSlaveNodes(N, U, mat=True)
    # plot handles
    rFig, rAx = plt.subplots()
    iFig, iAx = plt.subplots()
    # axis setup
    for a in [rAx, iAx]:
        a.set_aspect('equal')
        a.set_xlabel(r'$x_1$')
        a.set_ylabel(r'$x_2$')
    rAx.set_title(r'$\Re (u)$')
    iAx.set_title(r'$\Im (u)$')
    
    # if more levels than meshpoints in each dimension, could be difficult! Throw a warning
    if levels >= N:
        print('Number of contour levels exceeds or equals N!')
    # make contour plots
    rCon = rAx.contourf(x, y, np.real(u), levels=levels)
    iCon = iAx.contourf(x, y, np.imag(u), levels=levels)
    # make colourbars
    rFig.colorbar(rCon)
    iFig.colorbar(iCon)
    
    return [rFig, rAx], [iFig, iAx]

def PlotFn3D(N, U):
    '''
    Create 3D-surface plots for the eigenfunction U that is passed in.
    INPUTS:
        N: int, number of meshpoints in each dimension
        U: ((N-1)**2,) complex, column vector of eigenfunction values at (non-slave) meshpoints
    OUTPUTS:
        Two lists of the form [fig, ax] for the handles of the matplotlib figures,
        the first containing the plot of the real part of U and 
        the second containing the plot of the imaginary part of U.    
    '''

    # gridpoints used
    x = y = np.linspace(0,1, num=N)
    X, Y = np.meshgrid(x,y)
    # restore "slave" meshpoints
    u = InsertSlaveNodes(N, U, mat=True)
    # plot handles
    rFig = plt.figure(); rAx = rFig.add_subplot(111, projection='3d')
    iFig = plt.figure(); iAx = iFig.add_subplot(111, projection='3d')
    # axis setup
    for a in [rAx, iAx]:
        a.set_xlabel(r'$x_1$')
        a.set_ylabel(r'$x_2$')
        a.set_zlabel(r'$u(x)$')
        a.set_xlim([0.,1.])
        a.set_ylim([0.,1.])
    rAx.set_title(r'$\Re (u)$')
    iAx.set_title(r'$\Im (u)$')
    
    # make contour plots
    rCon = rAx.plot_surface(X, Y, np.real(u), cmap=cm.viridis, linewidth=0)
    iCon = iAx.plot_surface(X, Y, np.imag(u), cmap=cm.viridis, linewidth=0)
    # make colourbars
    rFig.colorbar(rCon)
    iFig.colorbar(iCon)
    
    return [rFig, rAx], [iFig, iAx]

# visual check on what Python thinks the eigenvalues of the FDM actually are
def PlotEvals(wVals, N=0, autoPlotWidow=False):
    '''
    Creates a (scatter) plot on the 2D complex plane of the distribution of "eigenvalues" found by the FDM method.
    INPUTS:
        wVals: (n,) complex, eigenvalues found from the FDM solve
        N: (optional) int, the number of meshpoints in each dimension, will be inferred if not provided explicitly
    OUTPUTS:
        fig, ax: matplotlib figure handles, contains a 2D scatter plot of the eigenvalues in the complex plane
    If N is not provided, the function assumes that wVals contains all the eigenvalues for it's mesh, IE that 
    len(wVals) = n = (N-1)^2.
    '''
    
    # can recover mesh size from number of eigenvalues found if we need to
    if N==0:
        N = np.sqrt(len(wVals)) + 1
    # otherwise, N is set for us by the user for cases when only a selection of the e'vals are to be plotted
    
    fig, ax = plt.subplots()
    #ax.set_aspect('equal')
    ax.set_title(r'Eigenvalues in the complex plane (N=%d)' % (N))
    ax.set_xlabel(r'$\Re(z)$')
    ax.set_ylabel(r'$\Im(z)$')
    ax.scatter(np.real(wVals), np.imag(wVals), marker='x', s=2, c='black')
    
    return fig, ax

def SmallestValuesIndex(a, k=19):
    '''
    Return the indices of the k smallest values in the array a. By default, k is the minimum of 19 and len(a).
    INPUTS:
        a: (n,) float, values to search
        k: (optional) int, the indices corresponding to the k-smallest values in a will be returned
    OUTPUTS:
        inds: (k,) int, the indices corresponding to the k-smallest values in a
    Note that inds will not be sorted in ascending order, in general a[inds[i-1]] !< a[inds[i]].
    '''
    if k>len(a):
        return np.argpartition(a, len(a))
    return np.argpartition(a, k)[:k]

# COMMAND LINE SCRIPT
if __name__=='__main__':
	
	parser = argparse.ArgumentParser(description='FDM assembly and eigenvalue solve, for the Cross-in-plane geometry.')
	parser.add_argument('-N', type=int, help='<Required> Number of meshpoints in each dimension.', required=True)
	parser.add_argument('-t1', default=0.0, type=float, help='QM_1 will be set to this value multiplied by pi.')
	parser.add_argument('-t2', default=0.0, type=float, help='QM_2 will be set to this value multiplied by pi.')
	parser.add_argument('-a3', default=0.0, type=float, help='Coupling constant value at v_3.')
	parser.add_argument('-l', action='store_true', help='Print log to screen whilst constructing FDM.')
	parser.add_argument('-sEvals', action='store_false', help='Computed eigenvalues will NOT be saved - overwrites default behaviour.')
	parser.add_argument('-sEvecs', action='store_true', help='Computed eigenvectors will not be saved.')
	
	args = parser.parse_args()
	# number of meshpoints
	# this is defined up here to save an argument being passed to M2C every time we want to use it
	N = args.N
	if N%2==0:
	    print('N = %d is not even, using N=%d instead' % (N, N+1))
	    N += 1
	else:
		print('Read N=%d' % (N))
	# mesh width
	h = 1./(N-1)
	# quasi-momentum value
	theta = np.array([args.t1, args.t2], dtype=float)
	print('Quasi-momentum set as: ', theta, '\pi')
	theta *= np.pi
	# coupling constant at v_3
	alpha3 = args.a3
	print('Read alpha_3 as: ', alpha3)
	# toggle whether log will write to screen
	logOn = args.l
	# settings for preserving the values computed
	saveEvals = args.sEvals
	saveEvecs = args.sEvecs
	
	# gridpoint placements
	x = y = np.linspace(0.,1.,num=N)
	
	# Function to change column <- matrix indices
	def M2C(i,j):
	    '''
	    Provides the column index in U for the gridpoint u_{i,j}.
	    INPUTS:
	        i,j: int, gridpoint indices
	    OUTPUTS:
	        c: int, index such that U[c] = u_{i,j}
	    '''
	    
	    return j + (N-1)*i

	# Begin assembly script here.
	sizeFDM = (N-1)*(N-1)
	# Initalise FDM
	FDM = np.zeros((sizeFDM,sizeFDM), dtype=complex)
	
	# Assemble row entries for each region
	for r in range(4):
	    RegionalFDM(FDM, r+1, N, theta, log=logOn)
	
	# Assemble row entries for each horizontal edge
	for he in [23,34]:
	    HorEdgeFDM(FDM, he, N, theta, log=logOn)
	
	# Assemble row entries for each vertical edge
	for ve in [31,53]:
	    VerEdgeFDM(FDM, ve, N, theta, log=logOn)
	    
	# Assemble row entry for v_3
	v3FDM(FDM, alpha3, N, log=logOn)
	
	# Is it Hermitian?
	if IsHermitian(FDM):
	    print('FDM is Hermitian')
	elif IsSymmetric(FDM):
	    print('FDM is symmetric')
	else:
	    print('FDM is NOT Symmetric nor Hermitian, see FDM - FDM.H below:')
	    diff = FDM - Herm(FDM)
	    print('Max real difference: ', np.max(np.abs(np.real(diff))))
	    print('Max imag difference: ', np.max(np.abs(np.imag(diff))))
	
	# Compute e'values and e'vectors.
	# NOTE: if alpha_3 is zero, will need to insert B(N) for a generalised eigenvalue problem
	# Otherwise, we can just pass FDM into eig
	print('Eigenvalue solving, this may take a while...', end='')
	if np.abs(alpha3)<=1e-8:
		wVals, wVecs = eig(FDM, B(N))
	else:
		wVals, wVecs = eig(FDM)
	print(' finished')
	
	# reconstruct every solution and check that the result is periodic (slave boundaries have been matched correctly)
	p = 0; lrFail = 0; tbFail = 0
	for w in range(len(wVals)):
	    wV = InsertSlaveNodes(N, wVecs[:,w], mat=True)
	    tf = CheckPeriodicBCs(N, wV)
	    if (tf[0] and tf[1]):
	        # this solution is fine upon reconstruction
	        p += 1
	    elif tf[0]:
	        # lr boundary fine, tb boundary not periodic
	        tbFail += 1
	        print('E-vec at index %d not periodic Top <-> Bottom' % (w))
	    elif tf[1]:
	        # tb boundary fine, lr boundary not periodic
	        lrFail += 1
	        print('E-vec at index %d not periodic Left <-> Right' % (w))
	    else:
	        # no periodicity on either solution!
	        lrFail += 1
	        tbFail += 1
	        print('E-vec at index %d not periodic in either direction' % (w))

	# clear infs and NaN evals
	eVals, tf = Purge(wVals)
	eVecs = wVecs[:,tf]
	realTF = RealEvalIndices(eVals, tol=1e-8)

	print('----- Analysis ----- \n #E-vals found: %d' % (len(wVals)))
	print('#Inf/NaN values: %d' % (len(wVals)-len(eVals)))
	print('#Real eigenvalues: %d' % (np.sum(realTF)))
	print('Reconstructed solutions: Periodic %d / LR fail %d / TB fail %d / Both fail %d' %
	      (p,lrFail,tbFail,lrFail+tbFail+p-len(wVals)))
	print('-----')

	# save the eigenvalues and a record of the quasi-momentum. Also throw in N for good measure
	if saveEvals:
	    saveStr = './EvalDump/EvalsN-' + str(N) + '_' + datetime.today().strftime('%Y-%m-%d-%H-%M') + '.npz'
	    np.savez(saveStr, eVals=eVals, qm=theta, N=N)
	    print('Saved eigenvalues to file:', saveStr)
	else:
		print('sEvals flagged: computed eigenvalues HAVE NOT been saved')
	if saveEvecs:
	    saveStr = './EvalDump/EvecsN-' + str(N) + '_' + datetime.today().strftime('%Y-%m-%d-%H-%M') + '.npz'
	    np.savez(saveStr, eVecs=eVecs, qm=theta, N=N)
	    print('Saved eigenvalues to file:', saveStr)