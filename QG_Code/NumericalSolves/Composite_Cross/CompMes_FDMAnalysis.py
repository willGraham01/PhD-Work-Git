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

from CompMes_FDM import RealEvalIndices

#%% Extract information from files and setup data stores

def ReadEvals_FDM(fname, funsToo=False):
    '''
    Extracts the eigenvalue information saved by the variational problem solver,
    and optionally the eigenvalues too
    INPUTS:
        fname: str, filename or path to file containing eigenvalue output
        funsToo: bool, if True then the eigenfunctions will also be extracted for us to examine
    OUTPUTS:
        valInfo: (nPts^2, 2+nEvals) complex, rows contain (in order) theta1/pi, theta2/pi, omega_1, ... , omega_n.
        efs: (nEvals*nPts^2, N*N) complex, row i*nEvals+j contains the eigenvector for the eigenvalue at valInfo[i,2+j]
    '''
    
    valInfo = np.loadtxt(fname, delimiter=',', dtype=complex)
    valInfo[:,:2] *= pi
    
    # if we want the eigenfunctions too...
    if funsToo:
        try:
            # try finding where in the string the file extension begins,
            # and inserting funcs there to get the filename for the functions
            extensionIndex = fname.rfind('.')
            vecFilename = fname[:extensionIndex] + '-funcs' + fname[extensionIndex:]
        except:
            # if you couldn't find where the file extension is,
            # just append to the end of the string, it's in there
            vecFilename = fname + '-funcs'
        # load function data too
        efs = np.loadtxt(vecFilename, delimiter=',', dtype=complex)
    else:
        efs = None

    return valInfo, efs

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

def GetBand(band, evs, tol=1e-8):
    '''
    Extracts the band-th eigenvalue from each row in evs.
    Converts to real numbers, and throws an error up if there is an eigenvalue which cannot be cast to a real number.
    INPUTS:
        band: int, band number to extract eigenvalues of (starting from band 1)
        evs: (nRuns,2+n) float, rows corresponding to nRun eigenvalue runs in which n eigenvalues were computed
        tol: float, eigenvalues with an imaginary component >tol are flagged bad and removed
    OUTPUTS:
        bandInfo: (nRuns,3) float, columns 0,1 are the quasimomentum values corresponding to the eigenvalue at column 2.
    '''
    
    # this is the number of eigenvalues per QM that was computed
    nEvals = np.shape(evs)[1] - 2
    # check that we have information on this band
    if band>nEvals:
        raise ValueError('Want band %d but only have information up to band %d' % (band, nEvals))
        
    # check for imaginary eigenvalues
    tf = RealEvalIndices(evs[:,1+band], tol=tol)
    if not tf.all():
        # at least one eigenvalue was removed for being too imaginary, report this
        print('Removed %d bad eigenvalues from band %d' % (np.shape(evs)[0]-np.sum(tf),band))
    # filter out bad points, then taking real part is fine
    goodEvs = evs[tf,:]
    bandInfo = np.real(goodEvs[:,(0,1,1+band)])
    return bandInfo

def LoadAllFromKey(searchPath, funsToo=False):
    '''
    Loads all eigenvalue information from the given search-path into a single eigenvalue array
    INPUTS:
        searchPath: str, string to expand and load eigenvalue information from
    OUTPUTS:
        allEvals: (nRuns,2+nEvals) complex, eigenvalue information arrays
    '''
    
    allFiles = glob.glob(searchPath, recursive=False)
    evList = []
    # only return the eigenfunctions if they were asked for
    if funsToo:
        fList = []
        for fname in allFiles:
            e, F = ReadEvals_FDM(fname, funsToo=True)
            # order the eigenvalues into ascending order by real part
            # we need to save this order so that the eigenfunctions are correctly re-ordered
            order = np.argsort(np.real(e[:,2:]))
            # now sort the eigenvalues
            e[:,2:] = np.sort_complex(e[:,2:])
            # record e'val array
            evList.append(e)
            # now order the eigenfunctions properly
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
            e, _ = ReadEvals_FDM(fname, funsToo=False)
            # order the eigenvalues into ascending order by np.sort_complex
            e[:,2:] = np.sort_complex(e[:,2:])
            # record e'val array
            evList.append(e)
        # combine e'value arrays
        allEvals = AppendEvalRuns(evList)        
    return allEvals, allVecs

#%% Data analysis/ handling

def Multiplicities(evs, rtol=1e-5, atol=1e-8):
    '''
    Returns a list of the spectral bands, but accounting for possible multiplicity in the eigenvalues that we have computed.
    INPUTS:
        ev: eigenvalue information, output of loading the eigenvalues from files
        rtol, atol: see np.isclose().
    OUTPUTS:
        bands: list of numpy arrays, bands[i][j,:] = [theta0, theta1, omega_j]
    '''
    
    # initialise band storage
    bands = []
    # how many eigenvalues are there for us to look at?
    nEvals = np.shape(evs)[1] - 2
    if nEvals==1:
        # only one band computed, so no "repeats" to find. Return a filtered band 1
        tf = RealEvalIndices(evs[:,2], tol=1e-5)
        bands.append(np.real(evs[tf,:]))
        if np.sum(np.logical_not(tf))>=1:
            print('Removed %d bad eigenvalues from band 0' % np.sum(np.logical_not(tf)))
        return bands
    else:
        # if there are more than 1 eigenvalue per QM, we can look for repeated eigenvalues
        areClose = np.zeros_like(evs[:,2:], dtype=bool)
        # areClose[i,b] = True implies row i, values in band b and b-1 are close
        for b in range(1,nEvals):
            areClose[:,b] = np.isclose( evs[:,2+(b-1)], evs[:,2+b] )
        # perpare outputs...
        filteredEvs = np.copy(evs)
        # for those rows that had repeated eigenvalues, go over each one in sequence, checking for repeats and identifying which band the repeats belong to
        for qm in range(np.shape(evs)[0]):
            # there's nothing to do if there weren't any repeats
            if areClose[qm,:].any():
                # find the first "set" of eigenvalues that could all be the same
                flags = np.nonzero(areClose[qm,:])[0]
                # if flags[n]+1!=flags[n+1], this is a new set of eigenvalues that could be the same
                evalGroups = []
                # evalGroups[i] is a list, [start,end], where evs[qm,2+start] through to evs[qm,2+end] (inclusive) are repeats of the same eigenvalue
                # the number of entries in evalGroups is equal to the number of separate groups of eigenvalues that might be repeats
                if flags.shape[0]==1:
                    # only one flag, implying only one pair of evals that could be the same
                    # return this information
                    evalGroups.append([flags[0]-1,flags[0]])
                else:
                    # possibly more than one group of eigenvalues, time to work out the groups
                    start = 0
                    n = 0
                    while start<flags.shape[0]:
                        if flags[n]+1 != flags[n+1]:
                            # flags not consecutive, have moved onto a new group of eigenvalues
                            evalGroups.append([flags[start]-1,flags[n]])
                            start = n+1
                            if start==flags.shape[0]-1:
                                # new start point is the end of flags, break the loop after recording final flag set
                                evalGroups.append([flags[-1]-1,flags[-1]])
                                start = flags.shape[0]
                        elif n==flags.shape[0]-2:
                            # flags ARE consecutive, but we've reached the end of the array
                            evalGroups.append([flags[start]-1,flags[-1]])
                            start = flags.shape[0]
                        n += 1
                # now using evalGroups, we need to decide which bands the eigenvalues belong to!
                print('qm row %d, evalGroups are' % qm, evalGroups)
                for group in evalGroups:
                    # group[0] is the lowest band the repeat appears in
                    # group[1] is the highest band the repeat appears in
                    
                    # check 1: what is the min. distance from you to any other eigenvalue in these bands?
                    # you should be close to the other values in the bands
                    dists = np.zeros((np.shape(evs)[0]-1,group[1]-group[0]+1),dtype=complex)
                    for i,g in enumerate(range(group[0], group[1]+1)):
                        dists[:,i] = np.delete(evs[:,2+g], qm) - evs[qm,2+g]
                    # dists[i] is a column vector of all the differences between this repeated eigenvalue and the other eigenvalues in band groups[0]+i
                    minDists = np.min(np.abs(dists), axis=0)
                    closestBand = np.argmin(minDists)
                    # groups[0]+closestBand is the band that this eigenvalue should belong to
                    # as such, set all the other entries in filteredEvs where this eigenvalue is repeated to be -1, for removal later
                    for i in range(group[0], group[1]+1):
                        if i-group[0]!=closestBand:
                            filteredEvs[qm,2+i] = -1.
        # filteredEvs now contains -1s at the entries that corresponded to repeated eigenvalues
        # from this, we can extract the bands
        for b in range(nEvals):
            tf = np.real(filteredEvs[:,2+b])>=0.
            bvs = filteredEvs[:,(0,1,2+b)]
            bvs = bvs[tf,:]
            tf = RealEvalIndices(bvs[:,2], tol=1e-5)
            bvs = np.real(bvs[tf,:])
            bands.append(bvs)
    return bands

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
    ax.set_title(r'Spectral bands (FDM)')
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

    parser = argparse.ArgumentParser(description='Plotting and results analysis script for the Cross-in-Plane geometry eigenvalues, computed via the Finite Difference Scheme.')
    parser.add_argument('path_to_file', type=str, help='<Required> Path to file containing results of FDM eigenvalue solve.')
    parser.add_argument('-fOut', default='./FDM_Results/', type=str, help='<Default .> File location to save plot outputs to.')
    parser.add_argument('-e', action='store_true', help='If passed,  eigenvalues corresponding to extreme QM values [0,0] and [-pi,-pi] will be highlighted in spectral plot.')
    parser.add_argument('-i', action='store_true', help='If passed,  eigenvalues corresponding to extreme QM values [-pi,0] and [0,-pi] will be highlighted in spectral plot.')
    parser.add_argument('-l', action='store_true', help='If passed, plots bands using lines between extreme eigenvalues, rather than scatter. .pdf output will be smaller in size, but should only be used if the spectrum is known to consist of (or previous plots demonstrate that it consists of) continuous bands.')
    parser.add_argument('-b', action='store_true', help='Create and save plots of the eigenvalues as functions of the QM.')
    parser.add_argument('-maxB', const=5, nargs='?', type=int, help='Only create plots for bands 0 through to maxB-1. Defaults to 5 if no value is passed with the flag.')
    parser.add_argument('-multi', action='store_true', help='If passed, filter the eigenvalues to remove those with multiplicity greater than 1.')
    parser.add_argument('-t', default='surf', choices=['contour', 'heat', 'scatter', 'surf'], help='Type of plot to make for eigenvalues against QM. Options are')

    # extract input arguments and get the setup ready
    args = parser.parse_args() 
 
    # get timestamp for saving plots later
    now = args.fOut + 'FDM_' + datetime.today().strftime('%Y-%m-%d-%H-%M')
    
    # load the data that we have been passed
    searchPath = args.path_to_file #'nPts51-N251-10evals.csv'
    
    allEvals, allEvecs = LoadAllFromKey(searchPath, funsToo=False)
    
    # this is the number of bands we tried to compute
    nEvals = np.shape(allEvals)[1] - 2
    # get all the bands, and put them into a list
    if args.multi:
        bands = Multiplicities(allEvals)
    else:    
        bands = []
        for n in range(nEvals):
            bands.append(GetBand(n+1, allEvals, tol=1e-5))
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