#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 09:07:24 2022

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
from numpy import pi

import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
from mpl_toolkits.axes_grid1 import make_axes_locatable

from CompMes_VarProb import GlobalVarProbSolve, Real2Comp, Poly2D

#%%

#%% Main-line script

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='VP convergence rate test: checks the speed of convergence to analytic eigenfunctions, by successively increasing the number of basis funtions M used in the approximation. Produces plots showing the rate of convergence.')
    parser.add_argument('-Mmax', default=11, type=int, help='<Default 11> Maximum highest order of approximation (M) to go to in convergence test.')
    parser.add_argument('-Mmin', default=6, type=int, help='<Default 6> Minimum highest order of approximation (M) to use in convergence test.')
    parser.add_argument('-Mstep', default=1, type=int, help='<Default 1> Incriment for M from Mmin to Mmax.')
    parser.add_argument('-n', default=1, type=int, help='<Default 1> Index n in analytic solution.')
    parser.add_argument('-m', default=1, type=int, help='<Default 1> Index m in analytic solution.')
    parser.add_argument('-nGiveUp', default=20, type=int, help='<Default 10> Number of eigenvalues to compute before giving up on finding the analytic solution.')
    parser.add_argument('-nIts', default=10, type=int, help='<Default 10> Number of iterations for BasinHopping.')
    parser.add_argument('-nInner', default=2500, type=int, help='<Default 2500> Number of iterations for minimise within BasinHopping step.')
    parser.add_argument('-fOut', default='./', type=str, help='<Default .> File location to save outputs (all plots and numerical data) to.')
    parser.add_argument('-sd', action='store_true', help='If passed, writes the error and M values out to an .npz file.')
    parser.add_argument('-plotBest', action='store_true', help='If passed, creates a plot of the best numerical approximation to the eigenfunction that was computed.')
    parser.add_argument('-load', default='', type=str, help='If provided with the path to an .npz file, the data to be plotted will be loaded from this file rather than computed by the script. The .npz file should contain two arrays of the same length, MVals (ints) and sqErrors (float). Passing a filename will suppress the use of Mmax, Mmin, Mstep, n, m, nEvals, plotBest (as the eigenfunctions will not be computed) and sd.')
    
    args = parser.parse_args()    
    # check compatibility
    if args.n<1 or args.m<1:
        raise ValueError('(n,m) = (%d,%d) cannot be less than 1' % (args.n,args.m))
        sys.exit(1)

    # get timestamp for saving plots later
    now = args.fOut + 'VP_' + datetime.today().strftime('%Y-%m-%d-%H-%M')
    # define the analytic constants here
    n = args.n
    m = args.m
    # derive values from n,m
    lbda = (n*n + m*m)*pi*pi
    omega = np.sqrt(lbda)    
    # set the quasi-momentum to the fixed value
    theta = -np.pi * np.ones((2,), dtype=float)
    if n%2==0:
        theta[0] = 0.
    if m%2==0:
        theta[1] = 0.
    # decide on the values of M that you want to use
    MVals = np.arange(args.Mmin,args.Mmax+1,args.Mstep,dtype=int)
    # how many eigenvalues and eigenvectors to find before giving up with finding the analytic solution.
    # Since we have to find eigenvalues in order, if we don't exceed the analytic eigenvalue after this many
    # solves, we just give up with this value of M and work with what we've got
    nGiveUp = args.nGiveUp
    # do you want to plot the *best* (in terms of error) approximate solution?
    plotBest = args.plotBest; nPts = 250; levels = 10
    # create error in eigenvalue plots?
    errorPlots = True
    # storage for the eigenvalues and the eigenvectors that were found will be in a dictionary, whose keys match the values in MVals
    # each value will be a list containing the (square) eigenvalue found and the corresponding eigenfunction approximation
    results = {}
    
#%%
    # was the data provided in a file?
    if args.load:
        # filename was provided, load this data instead
        loadedData = np.load(args.load)
        NVals = loadedData['MVals']
        sqErrors = loadedData['sqErrors']
    else:
        # we need to compute the data ourselves
        # perform runs varying M over MVals to obtain the information we need
        for M in MVals:
            print('   SOLVING M=%d    ' % M)
            # the plan now is to solve the minimisation VP problem until either we find an eigenvalue that
            # exceeds the analytic eigenvalue, or until a convergence error
            prevUs = []
            currEv = 0.
            bestInd = 0.
            nEvalsFound = 0
            while nEvalsFound<nGiveUp and currEv<lbda:
                # whilst we have not computed nGiveUp evs or exceeded the analytic eigenvalue
                uStore, sqEv, conFlag = GlobalVarProbSolve(M, 1, theta, prevUs=prevUs, nIts=args.nIts, nIts_inner=args.nInner, lOff=False)
                # store eigenfunction found for next solves
                prevUs.append( Poly2D(theta, Real2Comp(uStore[0,:])) )
                # this is the current eigenvalue
                currEv = sqEv[0]
                if conFlag!=-1:
                    print('Convergence fail at n=%d, aborting remaining solves' % nEvalsFound)
                    break
                else:
                    # continue searching until be exceed the analytic eigenvalue
                    nEvalsFound += 1
            # catch fails when we didn't get ANY usable eigenvalues
            if nEvalsFound==0:
                # didn't get any usable eigenvalues
                continue
            else:
                # now determine which eigenfunction is "closest" to the analytic solution
                reliableFns = prevUs[0:nEvalsFound]
                sqEvals = np.zeros((len(reliableFns),),dtype=float)
                for i,p in enumerate(reliableFns):
                    sqEvals[i] = p.lbda
                # index of best approximation
                bestInd = np.argmin(np.abs(sqEvals-lbda))
                # append result to dictionary
                results[M] = prevUs[bestInd]
        # now we have results as a dictionary containing the best eigenfunction approximations for each M
        # compute sqErrors from this
        sqErrors = np.zeros( (len(list(results.keys())),), dtype=float )
        MVals = np.zeros( (len(list(results.keys())),), dtype=int )
        for i,M in enumerate(results.keys()):
            MVals[i] = M
            sqErrors[i] = np.abs( results[M].lbda - lbda )
    # this now means we have sqErrors, containing |omega_M^2 - lbda| for each M, and MVals, containing the
    # values of M
    # if requested, save data to file
    if args.sd:
        np.savez(now+'saveData', MVals=MVals, sqErrors=sqErrors)
#%% Create the plots that you asked for

    # plot the error as a function of M
    if errorPlots:
        erFig, erAx = plt.subplots(1)
        erAx.plot(MVals, sqErrors)
        erAx.set_xlabel(r'Highest order of approximating polynomial functions, $M$')
        erAx.set_ylabel(r'$\vert %d\pi^2 - \omega_M^2 \vert$' % (n*n+m*m))
        erAx.set_title(r'Error in eigenvalue against number of terms in approximation')
        erFig.savefig(now + '_Error.pdf', bbox_inches='tight')
            
    # plot the best (in terms of eigenvalue error) approximate eigenfunction
    if plotBest and (not args.load):
        # setup axes
        bestFig, bestAx = plt.subplots(ncols=2, sharey=False)
        
        # identify best eigenfunction from closest eigenvalue
        minM = MVals[np.argmin(sqErrors)] # this value of M had the eigenvalue closest to the true value
        uBest = Poly2D(theta, Real2Comp(results[minM][1]))

        # prepare plot data
        x = y = np.linspace(0,1,num=nPts)
        X, Y = np.meshgrid(x,y)
        Uvals = uBest.val(X,Y)
        # if more levels than meshpoints in each dimension, could be difficult! Throw a warning
        if levels >= nPts:
            print('Number of contour levels exceeds or equals nPts!')

        # axis display
        for a in bestAx:
            a.set_aspect('equal')
            a.set_xlabel(r'$x_1$')
            a.set_ylabel(r'$x_2$')
        bestAx[0].set_title(r'$\Re (u)$')
        bestAx[1].set_title(r'$\Im (u)$')
        bestFig.suptitle(r'Best approximating eigenfunction')
            
        # make contour plots (note: Uvals created with meshgrid so no need for transposes here)
        rCon = bestAx[0].contourf(X, Y, np.real(Uvals), levels=levels)
        iCon = bestAx[1].contourf(X, Y, np.imag(Uvals), levels=levels)
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
        print('Best approximation occurs when M=%d' % minM)
        bestFig.savefig(now + '_BestApprox.pdf', bbox_inches='tight')

    sys.exit(0)