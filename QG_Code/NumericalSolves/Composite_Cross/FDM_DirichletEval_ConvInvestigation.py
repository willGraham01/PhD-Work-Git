#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 11:29:54 2022

@author: will

The cross-in-plane geometry, and our Helmholtz equation wrt the composite measure on it, is satisfied by a subset of the eigenfunctions of the Dirichlet Laplacian on $\Omega$.

Observe that, for $\Omega=(0,1)^2$ and $\theta\in[-\pi,\pi)^2$, the eigenpairs of the Dirichlet Laplacian which solve the equation

\begin{align*}
    -\Delta_{\theta}u = \lambda u, \qquad u\vert_{\partial\Omega} = 0,
\end{align*}

are given by 

\begin{align*}
    u_{nm}(x,y) &= \mathrm{e}^{-\mathrm{i}\theta\cdot\mathbf{x}}\sin\left(n\pi x\right)\sin\left(m\pi y\right), \\
    \lambda_{nm} &= \left( n^2 + m^2 \right)\pi^2.
\end{align*}

In the event that both of the following hold:
- ( $n$ is even and $\theta_1=0$ ) or ( $n$ is odd and $\theta_1=-\pi$ ),
- ( $m$ is even and $\theta_2=0$ ) or ( $m$ is odd and $\theta_2=-\pi$ ),

then $u_{nm}$ and $\lambda_{nm}$ are also an eigenpair for our Helmholtz equation with respect to the composite measure.

This script looks to test whether the FDM solver is getting close to one of these eigenfunctions.
Specifically, we will look to test whether as we increase $N$ (the number of gridpoints used), the difference in the nearest eigenvalue and eigenfunction gradually gets closer to the analytic solution when $n=m=1$.

This provides us with the eigenfunction and eigenvalue

\begin{align*}
    U(x,y) &= \mathrm{e}^{\mathrm{i}\pi(x+y)}\sin\left(\pi x\right)\sin\left(\pi y\right), \\
    \lambda =: \omega^2 &= 2\pi^2, 
    \qquad \implies \omega \approx 4.442882938.
\end{align*}

Ammendum: later, we add the functionality to look for any solution indexed by n and m.

"""
import argparse
import sys

from datetime import datetime

import numpy as np
from numpy import pi, exp, sin

import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
from mpl_toolkits.axes_grid1 import make_axes_locatable

from CompMes_FDM import FDM_FindEvals, RealEvalIndices, PlotFn, InsertSlaveNodes

#%%

def Make_AnalyticU(n=1,m=1):
    '''
    Creates a function handle for the analytic solution U_{nm}(x,y) as defined above.
    INPUTS:
        n,m: int, index as above for the eigenfunction U. Default 1 and 1
    OUTPUTS:
        U(x,y): function, evaluates the analytic solution U at the coordinate pairs (x,y)
    '''
    
    # set the quasi-momentum to the correct value given the parity of n and m
    theta = -np.pi * np.ones((2,), dtype=float)
    if n%2==0:
        theta[0] = 0.
    if m%2==0:
        theta[1] = 0.    
    
    def U(x,y):
        '''
        Analytic solution U(x,y) as defined above.
        INPUTS:
            x,y: (n,) float, coordinate pairs to evaluate U at
        OUTPUTS:
            uVals: (n,n) float, value of U at the coordinate pairs
        '''
        
        uVals = exp(-1.j*(theta[0]*x + theta[1]*y)) * sin(n*pi*x) * sin(m*pi*y)
        return uVals
    
    return U

def PlotAnalytic(N, n=1, m=1, levels=10):
    '''
    Create a plot of the real and imaginary parts of the analytic solution using N gridpoints in each dimension.
    INPUTS:
        N: int, number of meshpoints in each dimension
        n,m: int, index as above for the eigenfunction U. Default 1 and 1
        levels: int, number of contour heights to draw
    OUTPUTS:
        uFig, uAx: matplotlib figure handles, heatmaps of the real and imaginary parts of the analytic solution
    '''
    
    # gridpoints used
    x = y = np.linspace(0,1, num=N)
    X, Y = np.meshgrid(x,y)
    U = Make_AnalyticU(n,m)
    # figure handles
    uFig, uAx = plt.subplots(ncols=2, sharey=False)
    # compute values of analytic solution at the gridpoints
    uVals = U(X,Y).reshape((N,N))
    # axis setup
    for a in uAx:
        a.set_aspect('equal')
        a.set_xlabel(r'$x_1$')
        a.set_ylabel(r'$x_2$')
    uAx[0].set_title(r'$\Re (u)$')
    uAx[1].set_title(r'$\Im (u)$')
    uFig.suptitle(r'Analytic Solution')
    
    # if more levels than meshpoints in each dimension, could be difficult! Throw a warning
    if levels >= N:
        print('Number of contour levels exceeds or equals N!')
    # make contour plots
    # remember matplotlib convention! X, Y, Z triples where Z has shape (M,N) where X of shape (N,) and Y of shape (M). Generated data with meshgrid, so should be auto-compatable...
    rCon = uAx[0].contourf(x, y, np.real(uVals), levels=levels)
    iCon = uAx[1].contourf(x, y, np.imag(uVals), levels=levels)
    
    # make colourbars
    rDiv = make_axes_locatable(uAx[0])
    rCax = rDiv.append_axes("right", size="5%", pad=0.05)
    uFig.colorbar(rCon, cax=rCax)
    iDiv = make_axes_locatable(uAx[1])
    iCax = iDiv.append_axes("right", size="5%", pad=0.05)
    uFig.colorbar(iCon, cax=iCax)
    # space things out a bit better
    uFig.tight_layout()
    uFig.subplots_adjust(top=1.3)
    
    return uFig, uAx
    
def TranslateFDM(u):
    '''
    Translates an eigenfunction from the FDM onto the domain of the VP. 
    The domains are identical, however the period cell was viewed in different ways: FDM has the central vertex at (1/2,1/2), whilst VP had the vertex at (0,0).
    As such, we need to move the entries of the vector u containing the FDM entries, so that we can view the two functions over the same reference domain.
    INPUTS:
        u: ((N-1)**2,) complex, eigenfunctions from FDM solve
    OUTPUTS:
        uShift: ((N-1)**2,) complex, eigenfunction u but now on the reference domain of VP
    '''
    
    # N from FDM; FDM has shape ((N-1)*(N-1),) due to the periodicity constraint
    N = int(np.sqrt( np.shape(u)[0] ) + 1)
    
    # FDM uses vertex at (1/2,1/2), whereas VP uses vertex at (0,0), so we need to translate by periodicity
    # this is compounded by the "repeated values" on the slave nodes in each of the arrays. So we don't want to re-insert the slave nodes just yet, as we want to translate first
    # for ease, let's move from our vector to the matrix representation
    matFDM = u.reshape(N-1,N-1)

    swaps = np.zeros_like(matFDM)
    swaps[0:N//2,0:N//2] = matFDM[N//2:,N//2:] # top right to bottom left
    swaps[N//2:,N//2:] = matFDM[0:N//2,0:N//2] # bottom right to top left
    swaps[0:N//2,N//2:] = matFDM[N//2:,0:N//2] # bottom right to top left
    swaps[N//2:,0:N//2] = matFDM[0:N//2,N//2:] # top left to bottom right
    
    uShift = swaps.reshape(((N-1)*(N-1),)) # replace vector of grid values
    return uShift

#%% Main-line script

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='FDM convergence rate test: checks the speed of convergence to analytic eigenfunctions, by successively increasing the number of gridpoints N in the mesh. Produces plots showing the rate of convergence.')
    parser.add_argument('-Nmax', default=100, type=int, help='<Default 100> Max. number of meshpoints to use in each dimension.')
    parser.add_argument('-Nmin', default=11, type=int, help='<Default 11> Min. number of meshpoints to use in each dimension. To ensure reasonable approximations, should be no less than 11.')
    parser.add_argument('-Nstep', default=4, type=int, help='<Default 4> Incriment for N number of meshpoints. Should be even.')
    parser.add_argument('-n', default=1, type=int, help='<Default 1> Index n in analytic solution.')
    parser.add_argument('-m', default=1, type=int, help='<Default 1> Index m in analytic solution.')
    parser.add_argument('-nEvals', default=5, type=int, help='<Default 5> Number of eigenvalues to compute near to analytic eigenvalue. Computing more "nearby" eigenvalues can help when the mesh is coarse.')
    parser.add_argument('-a3', default=0.0, type=float, help='<Default 0.> Coupling constant value at v_3.')
    parser.add_argument('-fOut', default='./', type=str, help='<Default .> File location to save outputs to.')
    parser.add_argument('-plotU', action='store_true', help='If passed, creates a plot of the analytic solution in the same directory as the convergence rate plots.')
    parser.add_argument('-plotBest', action='store_true', help='If passed, creates a plot of the best numerical approximation to the eigenfunction that was computed.')
    parser.add_argument('-logH', action='store_true', help='If passed, the convergence rate plot will be created with the log of the error against the mesh width h rather than the number of mesh points N.')
    parser.add_argument('-NoSparse', action='store_false', help='If passed, FDMs will be constructed without using sparse storage. Not recommended.')
    
    args = parser.parse_args()    
    # check compatibility
    if args.n<1 or args.m<1:
        raise ValueError('(n,m) = (%d,%d) cannot be less than 1' % (args.n,args.m))
        sys.exit(1)
    if args.Nstep%2!=0:
        raise ValueError('Nstep (%d) is odd - required to be even.' % (args.Nstep))
        sys.exit(1)
    if args.Nmin%2==0:
        raise ValueError('Nmin = %d must be odd.' % args.Nmin)
        sys.exit(1)

    # get timestamp for saving plots later
    now = args.fOut + datetime.today().strftime('%Y-%m-%d-%H-%M')
    # define the analytic constants here
    n = args.n
    m = args.m
    # derive values from n,m
    lbda = (n*n + m*m)*pi*pi
    omega = np.sqrt(lbda)    
    # set alpha3 to be zero, but you could change this if you wanted I guess. U would still be an analytic solution in this case since the value of the function at the vertex is 0 anyway...
    alpha3 = args.a3
    # set the quasi-momentum to the fixed value
    theta = -np.pi * np.ones((2,), dtype=float)
    if n%2==0:
        theta[0] = 0.
    if m%2==0:
        theta[1] = 0.
    # decide on the values of N that you want to use
    NVals = np.arange(args.Nmin,args.Nmax,args.Nstep,dtype=int)
    # tell the numerical scheme to find values near to the analytic eigenvalue first
    sigma = lbda
    # how many eigenvalues and eigenvectors to find. 1 should suffice since we're starting near the analytic answer, however if worried we can always find more just to check there's no weird behaviour going on
    nToFind = 5
    # do you want to plot all the eigenfunctions that you found too?
    plotEFs = False
    # do you want to plot the *best* (in terms of error) approximate solution?
    plotBest = args.plotBest; levels = 10
    # create error in eigenvalue plots?
    errorPlots = True
    # on log plots, use the mesh width h (True) or the number of gridpoints N (False)
    logWithH = args.logH
    # storage for the eigenvalues and the eigenvectors that were found will be in a dictionary, whose keys match the values in NVals
    # each value will be a list containing the (square) eigenvalue found and the corresponding eigenfunction approximation
    results = {}
    
#%% If you want to look at the analytic solution
    
    if args.plotU:
        uF, uA = PlotAnalytic(NVals[-1], n=n, m=m)
        saveStr = now + '_AnalyticSol.pdf'
        uF.savefig(saveStr, bbox_inches='tight')
    
#%% Perform runs with varying N to gather the data required

    # compute the approximation from the FDM
    for N in NVals:
        print('   SOLVING N=%d    ' % N)
        sqEvals, eVecs, _ = FDM_FindEvals(N, theta, alpha3, lOff=False, nEvals=nToFind, sigma=sigma, checks=False, saveEvals=False, saveEvecs=False, sparseSolve=args.NoSparse)
        if nToFind==1:
            # convert to arrays of different shape just to make conversions easier
            sqEvals = np.asarray([sqEvals])
            eVecs = eVecs.reshape((eVecs.shape[0],1))
        results[N] = [sqEvals, eVecs]
    
    # now analyse the results... first check that our eigenvalues were all real
    # this will store the difference from the correct eigenvalue for each N
    sqErrors = np.zeros_like(NVals, dtype=float)
    for i, N in enumerate(NVals):
        nRealEvals = RealEvalIndices(results[N][0])
        if np.sum(nRealEvals)==nToFind:
            # everything was real, so convergence is good
            # cast eigenvalues to floats safely
            results[N][0] = np.real(results[N][0])
        else:
            # at least one eigenvalue is bad...
            raise ValueError('%d non-real eigenvalues found for N=%d' % (np.sum(nRealEvals), N))
            
        # identify the eigenvalue found that is closest to the true lbda
        sqDiffs = lbda - results[N][0]
        bestEvalInd = np.argmin(np.abs(sqDiffs))
        # record the error in the eigenvalue
        sqErrors[i] = np.abs(sqDiffs[bestEvalInd])
        print('N=%d: closest eigenvalue is at index %d with difference of %.5e' % (N, bestEvalInd, sqErrors[i]))
        bestEF = results[N][1][:,bestEvalInd]
        # append the best index to the list so we can easily recover the correct eigenvalue later
        results[N].append(bestEvalInd)
        
        # plot the approximation to the eigenfunction for me, translating due to the domain
        if plotEFs:
            PlotFn(N, TranslateFDM(bestEF))
            
#%% Create the remaining plots that you asked for
    
    # plot the error as a function of N?
    if errorPlots:
        erFig, erAx = plt.subplots(1)
        erAx.plot(NVals, sqErrors)
        erAx.set_xlabel(r'Number of gridpoints in each dimension, $N$')
        erAx.set_ylabel(r'$\vert 2\pi^2 - \omega_N^2 \vert$')
        erAx.set_title(r'Error in eigenvalue against number of meshpoints')
        # now do it on log axes against the mesh width h = 1/(N-1)
        if logWithH:
            logFig, logAx = plt.subplots(1)
            logAx.plot(1/(NVals-1), np.log(sqErrors))
            logAx.set_xlabel(r'Mesh width, $h$')
            logAx.set_ylabel(r'$\log\left(\vert 2\pi^2 - \omega_N^2 \vert\right)$')
            logAx.set_title(r'Error in eigenvalue against number of meshpoints')
        else:
            logFig, logAx = plt.subplots(1)
            logAx.plot(NVals, np.log(sqErrors))
            logAx.set_xlabel(r'Number of gridpoints in each dimension, $N$')
            logAx.set_ylabel(r'$\log\left(\vert 2\pi^2 - \omega_N^2 \vert\right)$')
            logAx.set_title(r'Error in eigenvalue against number of meshpoints')
        erFig.savefig(now + '_Error.pdf', bbox_inches='tight')
        logFig.savefig(now + '_LogError.pdf', bbox_inches='tight')
            
    # plot the best (in terms of eigenvalue error) approximate eigenfunction
    if plotBest:
        bestFig, bestAx = plt.subplots(ncols=2, sharey=False)
        minN = NVals[np.argmin(sqErrors)] # this value of N had the eigenvalue closest to the true value
        uBest = TranslateFDM(results[minN][1][:,results[minN][2]])
        
        x = y = np.linspace(0,1, num=N)
        uBest = InsertSlaveNodes(minN, uBest, mat=True)
        for a in bestAx:
            a.set_aspect('equal')
            a.set_xlabel(r'$x_1$')
            a.set_ylabel(r'$x_2$')
        bestAx[0].set_title(r'$\Re (u)$')
        bestAx[1].set_title(r'$\Im (u)$')
        bestFig.suptitle(r'Best approximating eigenfunction')
        
        # if more levels than meshpoints in each dimension, could be difficult! Throw a warning
        if levels >= N:
            print('Number of contour levels exceeds or equals N!')
        # make contour plots
        # remember matplotlib convention! X, Y, Z triples where Z has shape (M,N) where X of shape (N,) and Y of shape (M) - need to transpose our data
        rCon = bestAx[0].contourf(x, y, np.real(uBest).T, levels=levels)
        iCon = bestAx[1].contourf(x, y, np.imag(uBest).T, levels=levels)
        # make colourbars
        rDiv = make_axes_locatable(bestAx[0])
        rCax = rDiv.append_axes("right", size="5%", pad=0.05)
        bestFig.colorbar(rCon, cax=rCax)
        iDiv = make_axes_locatable(bestAx[1])
        iCax = iDiv.append_axes("right", size="5%", pad=0.05)
        bestFig.colorbar(iCon, cax=iCax)
        # space things out a bit better
        bestFig.tight_layout()
        bestFig.subplots_adjust(top=1.3) #moves top title closer again
        print('Best approximation occurs when N=%d, eigenvalue index %d' % (minN, results[minN][2]))
        bestFig.savefig(now + '_BestApprox.pdf', bbox_inches='tight')

    sys.exit(0)