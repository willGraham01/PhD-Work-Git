#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 09:22:05 2021

@author: will

Python file and script containing methods to produce eigenvalue plots from the outputs of:
    CompMesProb_EvalFinder.py [variational problem solver]
"""

# sys.argv[1:] contains command line arguments. Index 0 is just the script name
import argparse
# for returning values to the command line
import sys
import glob
from datetime import datetime

import numpy as np
from numpy import pi

import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

from CompMes_VarProb import Poly2D, Real2Comp

#%% Extract information from files and setup data stores

def ReadEvals_VarProb(fname, funsToo=False):
    '''
    Extracts the eigenvalue information saved by the variational problem solver,
    and optionally the eigenvalues too
    INPUTS:
        fname: str, filename or path to file containing eigenvalue output
        funsToo: bool, if True then the eigenfunctions will also be extracted for us to examine
    OUTPUTS:
        valInfo: (3+n,) dataframe, rows contain (in order) convergence flag, theta1/pi, theta2/pi, omega_1, ... , omega_n.
        funcList: list of Poly2D, if funsToo is flagged, then a list of the eigenfunctions is returned. The list index [i*n+j] corresponds to the eigenfunction of the eigenvalue valInfo[i,3+j].
    '''
    
    valInfo = np.genfromtxt(fname, delimiter=',')
    # convert theta from -1 to 1 (as saved) to -pi to pi
    valInfo[:,1:3] *= pi
    funcList = []
    
    # if we want the eigenfunctions too...
    if funsToo:
        try:
            # try finding where in the string the file extension begins,
            # and inserting funcs there to get the filename for the functions
            extensionIndex = fname.rfind('.')
            funcFilename = fname[:extensionIndex] + '-funcs' + fname[extensionIndex:]
        except:
            # if you couldn't find where the file extension is,
            # just append to the end of the string, it's in there
            funcFilename = fname + '-funcs'
        # load function data
        funData = np.genfromtxt(funcFilename, delimiter=',')
        for row in range(np.shape(funData)[0]):
            # t1/pi, t2/pi are stored at column indices 1 and 2
            fTheta = pi * funData[row,1:3]
            # coefficients (as floats) are stored in funData[3:], need to cast to complex first
            funcList.append(Poly2D(fTheta, Real2Comp(funData[row, 3:])))
    return valInfo, funcList

def AppendEvalRuns(infoArrays):
    '''
    Vertically stacks the arrays infoArrays (given as a tuple or list).
    INPUTS:
        infoArrays: list or tuple, containing arrays to vertically stack atop each other
    OUTPUTS:
        stack: float array, array formed from vertical stacking
    '''
    
    if type(infoArrays)==list:
        stack = np.vstack(tuple(infoArrays))
    else:
        stack = np.vstack(infoArrays)
    return stack

def GetBand(band, evs, removeFailed=True):
    '''
    Extracts the band-th eigenvalue from each row in evs.
    INPUTS:
        band: int, band number to extract eigenvalues of (starting from band 1)
        evs: (N,3+n) float, rows corresponding to eigenvalue runs in which n eigenvalues were computed
        removeFailed: bool, if True then we don't extract eigenvalues for which convergence failed'
    OUTPUTS:
        bandInfo: (M,3) float, columns 0,1 are the quasimomentum values corresponding to the eigenvalue at column 2. M = N - number of failed convergences if removeFailed is True, otherwise = N.
    '''
    
    # this is the number of eigenvalues per QM that was computed
    N = np.shape(evs)[1] - 3
    # check that we have information on this band
    if band>N:
        raise ValueError('Want band %d but only have information up to band %d' % (band, N))
        
    bandInfo = evs[:,(1,2,2+band)]
    if removeFailed:
        # remove eigenvalue rows for which there was no convergence
        # nConv is equal to the first index n for which there was no convergence,
        # being -1 if everything converged.
        # Thus, we need noConv < 0 for band 1's value to be trusted,
        # noConv <1 for band 2's value, etc
        # account for saving and conversion errors by giving 0.5 leeway
        allGood = evs[:,0] < -0.5  # these indicies had no convergence issues
        goodToBand = evs[:,0] - band > -0.5 # these indicies had convergence issues, but after this band
        convInds = np.logical_or(allGood, goodToBand)
        #convInds = evs[:,0] < (band-1) - 0.5 
        # slice out bad eigenvalues
        bandInfo = bandInfo[convInds,:]
        print('Removed %d bad eigenvalues in band %d' % (np.shape(evs)[0]-np.sum(convInds), band))
    return bandInfo

def LoadAllFromKey(searchPath, funsToo=False):
    '''
    Loads all eigenvalue information from the given search-path into a single eigenvalue array
    INPUTS:
        searchPath: str, string to expand and load eigenvalue information from
        funsToo: bool, if True then load the eigenfunctions too, mainly for plotting purposes
    OUTPUTS:
        allEvals: (M,N+3) float, eigenvalue information arrays
    '''
    
    allFiles = glob.glob(searchPath, recursive=False)
    evList = []
    # only return the eigenfunctions if they were asked for
    if funsToo:
        fList = []
        for fname in allFiles:
            e, F = ReadEvals_VarProb(fname, funsToo=True)
            # order the eigenvalues into ascending order by real part
            # we need to save this order so that the eigenfunctions are correctly re-ordered
            # cheap way to do this is just to set all the non-converged eigenvalues to be the same value as the largest eigenvalue found in that row, +1 step
            
            nEvals = np.shape(e)[1] - 3
            nRows = np.shape(e)[0]
            # preliminary convergence checker
            for row in range(nRows):
                conFlag = int( e[row,0] ) # safe to cast since this is a float with hanging zeros
                if conFlag != -1:
                    # some of these eigenvalues are not converged and are masquarading as 0
                    # set these to be the maximum value in the row +1, then sort
                    e[row,3+conFlag:] = np.max(e[row,3:]) + 1.
            # now, we can just sort as normal, since non-converged eigenvalues will be sorted to the back of the array
            order = np.argsort(e[:,3:])
            # now sort the eigenvalues
            e[:,3:] = np.sort(e[:,3:])
            # record e'val array
            evList.append(e)
            # now order the eigenfunctions properly
            nEvals = np.shape(e)[1] - 3
            for row in order.shape[0]:
                # order[row,:] is the order that F[row*nEvals:(row+1)*nEvals,:] should be in
                F[row*nEvals:(row+1)*nEvals,:] = F[row*nEvals+order[row,:],:]
            # extend list of vector arrays
            fList.append(F)
        # combine e'value arrays
        allEvals = AppendEvalRuns(evList)
        # combine e'vctor arrays
        allVecs = AppendEvalRuns(fList)
    else:
        allVecs = None
        for fname in allFiles:
            e, _ = ReadEvals_VarProb(fname, funsToo=False)
            # order the eigenvalues into ascending order, accounting for possible non-convergences
            # cheap way to do this is just to set all the non-converged eigenvalues to be the same value as the largest eigenvalue found in that row, +1
            nEvals = np.shape(e)[1] - 3
            nRows = np.shape(e)[0]
            for row in range(nRows):
                conFlag = int( e[row,0] ) # safe to cast since this is a float with hanging zeros
                if conFlag != -1:
                    # some of these eigenvalues are not converged and are masquarading as 0
                    # set these to be the maximum value in the row +1, then sort
                    e[row,3+conFlag:] = np.max(e[row,3:]) + 1.
                e[row,3:] = np.sort(e[row,3:])
            # record e'val array
            evList.append(e)
        # combine e'value arrays
        allEvals = AppendEvalRuns(evList)        
    return allEvals, allVecs

#%% Plots and visualisation functions

def PlotEvals(evs, pType='scatter', title=''):
    '''
    Creates a 3D-scatter plot using the information about the eigenvalues passed in evs
    INPUTS:
        evs: (n,3) float, columns 0,1 contain the quasimomentum values, and column 2 contains the eigenvalue
        pType: str, one of 'scatter', 'surf', 'contour', 'heat'. Determines the type of plot that will be made.
        title: str, sets the figure title
    OUTPUTS:
        fig, ax: matplotlib axis handles, for a 3D scatter plot of the eigenvalues
    '''

    fig = plt.figure()
    # contour plots are 2D, all others are 3D
    if np.char.equal('contour', pType):
        ax = fig.add_subplot()
        # use tricontour since we don't have 2D array of data
        dataDisp = ax.tricontour(evs[:,0]/pi, evs[:,1]/pi, evs[:,2], cmap=plt.cm.viridis)
    elif np.char.equal('heat', pType):
        ax = fig.add_subplot()
        # use tricontourf since we don't have 2D array of data
        dataDisp = ax.tricontourf(evs[:,0]/pi, evs[:,1]/pi, evs[:,2], cmap=plt.cm.viridis)        
    else:
        ax = fig.add_subplot(projection='3d')
        ax.set_zlabel(r'$\omega$')
        if np.char.equal('scatter', pType):    
            dataDisp = ax.scatter(evs[:,0]/pi, evs[:,1]/pi, evs[:,2], c=evs[:,2], cmap=plt.cm.viridis)
        elif np.char.equal('surf', pType):
            dataDisp = ax.plot_trisurf(evs[:,0]/pi, evs[:,1]/pi, evs[:,2], cmap=plt.cm.viridis, linewidth=0, antialiased=False)
        else:
            raise ValueError('Unrecognised plot type %s, valid types are scatter, surf, contour, heat')
    
    # set axis labels and figure title
    ax.set_xlabel(r'$\frac{\theta_1}{\pi}$')
    ax.set_ylabel(r'$\frac{\theta_2}{\pi}$')
    # set the title if provided
    if title:
        ax.set_title(title)
    # create colourbar
    fig.colorbar(dataDisp, ax=ax)
        
    return fig, ax

def PlotBands(bands, markQMExtremes=False, intermediateExtremes=False, lines=False):
    '''
    Creates a plot of the spectral bands.
    INPUTS:
        bands: list of (3,) floats, each entry in the list is a set of coordinates (qm_1, qm_2, omega) where omega is the eigenvalue in this band corresponding to the QM value (qm_1,qm_2)
        markQMExtremes: bool, if True then the eigenvalues corresponding to QM = (0,0) and (-pi,-pi) will be highlighted on the plot.
        intermediateExtremes: bool, if True then eigenvalues corresponding to QM = (-pi,0) and (0,-pi) will be highlighted on the plot
        lines: bool, if True then spectral bands will be plotted as lines between the extreme values. This is recommended only for file compression when a large number of eigenvalues are to be plotted and saved to .pdf. Additionally, one should check that the bands appear to be continuous lines before toggling this option.
    OUTPUTS:
        fig, ax: matplotlib figure handles, containing a plot of the spectral bands
    '''
    
    fig, ax = plt.subplots(1)
    ax.set_xlabel(r'$\omega$')
    ax.set_title(r'Spectral bands (VP)')
    ax.set_ylabel(r'Band number')
    nBands = len(bands)
    # prepare y-axis band labels
    yticks = np.arange(nBands, dtype=float)/nBands
    yticklabels = np.arange(nBands, dtype=int)
    # plot each band, increasing the 'height' on the y-axis as we move to the next band
    if not lines:
        for b,band in enumerate(bands):
            ax.scatter(band[:,2], np.zeros_like(band[:,2])+yticks[b], marker='x', s=1, c='blue', label='Band %d' % b)
    else:
        # plot spectrum as lines rather than scatter
        for b,band in enumerate(bands):
            wData = np.array( [np.min(band[:,2]), np.max(band[:,2]) ] )
            ax.plot(wData, np.zeros_like(wData)+yticks[b], '-b', label='Band %d' % b)
    # set custom labels for axis
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    
    # mark QM extreme values if requested
    if markQMExtremes or intermediateExtremes:
        # extreme QM values are (0,0) and (-pi,-pi).
        # but we might also want to mark (-pi,0) and (0,-pi) too.
        for b,band in enumerate(bands):
            # find the eigenvalues corresponding to the symmetry points, and add them to the plot
            height = b/nBands
            # find where each component of the QM is either 0 or -pi
            z1 = np.isclose(band[:,0], 0);  z2 = np.isclose(band[:,1],0)
            p1 = np.isclose(band[:,0], -pi);  p2 = np.isclose(band[:,1], -pi)
            # find where both QM values are 0
            zzRow = np.logical_and(z1,z2)
            # find where both QM values are -pi
            ppRow = np.logical_and(p1,p2)
            # find where QM = (0,-pi)
            zpRow = np.logical_and(z1,p2)
            # find where QM = (-pi,0)
            pzRow = np.logical_and(p1,z2)
            # it is possible that the symmetry points had convergence fails, so only attempt to plot them if they were included in the array
            if zzRow.any() and markQMExtremes:
                ax.scatter(band[zzRow,-1], height, marker='o', c='red', s=2)
            elif markQMExtremes:
                print('Band %d, (0,0) not found' % b)
            if ppRow.any() and markQMExtremes:
                ax.scatter(band[ppRow,-1], height, marker='o', c='red', s=2)
            elif markQMExtremes:
                print('Band %d, (-pi,-pi) not found' % b)
            if zpRow.any() and intermediateExtremes:
                ax.scatter(band[zpRow,-1], height, marker='o', c='red', s=2)
            elif intermediateExtremes:
                print('Band %d, (0,-pi) not found' % b)
            if pzRow.any() and intermediateExtremes:
                ax.scatter(band[pzRow,-1], height, marker='o', c='red', s=2)
            elif intermediateExtremes:
                print('Band %d, (-pi,0) not found' % b)
    return fig, ax

#%% Command-line execution

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Plotting and results analysis script for the Cross-in-Plane geometry eigenvalues, computed via the Variational Problem (VP) solver.')
    parser.add_argument('path_to_file', type=str, help='<Required> Path to file containing results of VP eigenvalue solve.')
    parser.add_argument('-fOut', default='./CompMes_VP_Results/', type=str, help='<Default ./CompMes_VP_Results/> File location to save plot outputs to.')
    parser.add_argument('-e', action='store_true', help='If passed,  eigenvalues corresponding to extreme QM values [0,0] and [-pi,-pi] will be highlighted in spectral plot.')
    parser.add_argument('-i', action='store_true', help='If passed,  eigenvalues corresponding to extreme QM values [-pi,0] and [0,-pi] will be highlighted in spectral plot.')
    parser.add_argument('-l', action='store_true', help='If passed, plots bands using lines between extreme eigenvalues, rather than scatter. .pdf output will be smaller in size, but should only be used if the spectrum is known to consist of (or previous plots demonstrate that it consists of) continuous bands.')
    parser.add_argument('-b', action='store_true', help='Create and save plots of the eigenvalues as functions of the QM.')
    parser.add_argument('-maxB', const=5, nargs='?', type=int, help='Only create plots for bands 0 through to maxB-1. Defaults to 5 if no value is passed with the flag.')
    parser.add_argument('-multi', action='store_true', help='[CURRENTLY NOT SUPPORTED] If passed, filter the eigenvalues to remove those with multiplicity greater than 1.')
    parser.add_argument('-t', default='surf', choices=['contour', 'heat', 'scatter', 'surf'], help='Type of plot to make for eigenvalues against QM. Options are')

    # extract input arguments and get the setup ready
    args = parser.parse_args() 
 
    # get timestamp for saving plots later
    now = args.fOut + 'VP_' + datetime.today().strftime('%Y-%m-%d-%H-%M')
    
    # load the data that we have been passed
    searchPath = args.path_to_file #'nPts51-N251-10evals.csv'
    
    allEvals, allEvecs = LoadAllFromKey(searchPath, funsToo=False)
    
    # this is the number of bands we tried to compute
    nEvals = np.shape(allEvals)[1] - 3
    # get all the bands, and put them into a list
    if args.multi:
        raise ValueError('Error: -multi passed to VP analysis, but not implimented yet.')
        #bands = Multiplicities(allEvals)
    else:    
        bands = []
        for n in range(nEvals):
            bands.append(GetBand(n+1, allEvals, removeFailed=True))
    # if we don't want all the bands involved, truncate now
    if args.maxB:
        bands = bands[0:args.maxB]
        
#%% Now create the figures that were requested
    
    # spectral band plot
    specFig, specAx = PlotBands(bands, markQMExtremes=args.e, intermediateExtremes=args.i, lines=args.l)
    specFig.savefig(now + '_SpectralBands.pdf', bbox_inches='tight')
    plt.close(specFig)
    
    # if you want to plot the eigenvalues as functions of QM
    if args.b:
        for bi, b in enumerate(bands):
            f, a = PlotEvals(b, pType=args.t, title=r'$\omega$ values in band %d' % (bi))
            f.savefig(now + '_Band%d.pdf' % bi, bbox_inches='tight')
    
    # close all figure windows
    plt.close('all')
    sys.exit(0)