#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 16:41:37 2021

@author: will

This script handles comparisions between the outputs of CompMes_FDMSolve and CompMesVarProbSolve, including plotting of solutions and graphical representations of the eigenvalues (according to each method)
"""

import numpy as np
from numpy import pi

import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

from copy import copy

from datetime import datetime

from CompMes_VarProbAnalysis import LoadAllFromKey as VP_Load
from CompMes_VarProbAnalysis import PlotEvals
from CompMes_VarProbAnalysis import GetBand as VP_GetBand
from CompMes_VarProb import Poly2D, VarProb, Real2Comp, Comp2Real
from CompMes_VarProb import GlobalVarProbSolve

from CompMes_FDMAnalysis import LoadAllFromKey as FDM_Load
from CompMes_FDMAnalysis import GetBand as FDM_GetBand
from CompMes_FDM import PlotFn as FDM_PlotU
from CompMes_FDM import InsertSlaveNodes

#%% Plotting functionality for multi-plot figures

def FDM_InsertToPlot(ax, N, U, levels=10):
    '''
    Create heatmaps for the eigenfunction U that is passed in, placing them directly onto the axis that is passed in in ax.
	This function only parses eigenfunctions in the output format of the FDM solver, for the VP analogue, see VP_InsertToPlot.
    INPUTS:
		ax: matplotlib.pyplot axis (2,), the plots created will be directly placed onto this input
        N: int, number of meshpoints in each dimension
        U: ((N-1)**2,) complex, column vector of eigenfunction values at (non-slave) meshpoints
        levels: (optional) int - default 10, number of contour levels to pass to contourf.
    OUTPUTS:
        ax0Con, ax1Con: contourf objects, with 0 being for the real part and 1 being for the imag part.
		This will enable colourbar creation in the wrapping program.
    '''
    
    # gridpoints used
    x = y = np.linspace(0,1, num=N)
    # restore "slave" meshpoints
    u = InsertSlaveNodes(N, U, mat=True)
    # axis setup
    for a in ax:
        a.set_aspect('equal')
        a.set_xlabel(r'$x_1$')
        a.set_ylabel(r'$x_2$')
    ax[0].set_title(r'$\Re (u)$')
    ax[1].set_title(r'$\Im (u)$')
    
    # if more levels than meshpoints in each dimension, could be difficult! Throw a warning
    if levels >= N:
        print('Number of contour levels exceeds or equals N!')
    # make contour plots
	# remember matplotlib convention! X, Y, Z triples where Z has shape (M,N) where X of shape (N,) and Y of shape (M) - need to transpose our data
    ax0Con = ax[0].contourf(x, y, np.real(u).T, levels=levels)
    ax1Con = ax[1].contourf(x, y, np.imag(u).T, levels=levels)
    
    return ax0Con, ax1Con

def VP_InsertToPlot(ax, U, nPts=250, levels=10):
	'''
	    Create heatmaps for the eigenfunction U that is passed in, placing them directly onto the axis that is passed in in ax.
	This function only parses eigenfunctions in the output format of the VP solver, for the FDM analogue, see FDM_InsertToPlot.
	    INPUTS:
	ax: matplotlib.pyplot axis (2,), the plots created will be directly placed onto this input
	        U: (M*M,) complex, column vector of coefficients in the representation of U
	nPTs: int, number of gridpoints to use in each dimension for plot - default is 250
	        levels: (optional) int - default 10, number of contour levels to pass to contourf.
	    OUTPUTS:
	        ax0Con, ax1Con: contourf objects, with 0 being for the real part and 1 being for the imag part.
	This will enable colourbar creation in the wrapping program.
	'''
	
	for a in ax:
		a.set_aspect('equal')
		a.set_xlabel(r'$x_1$')
		a.set_ylabel(r'$x_2$')
	# create Poly2D object to quickly evaluate eigenfunction
	# note that theta is irrelevant for computing the values of the approximation, so we can get away with just passing in 0's here
	# note that we need to shave off the QM values stored at the front
	P = Poly2D(np.zeros((2,)), U[2:])

	# now create plots...
	x = y = np.linspace(0,1,num=nPts)
	X, Y = np.meshgrid(x,y)
	u = P.val(X,Y)
	ax0Con = ax[0].contourf(X, Y, np.real(u), levels=levels)
	ax1Con = ax[1].contourf(X, Y, np.imag(u), levels=levels)
	return ax0Con, ax1Con

#%% Compare eigenfunctions from the two methods

def PlotNonSimpleEVs(repInfo, bands, bandEFs, nN, nMultPlots=10, VP=True, saveTo='', saveType='.pdf'):
	'''
	Given information about the repeated eigenvalues, and the eigenvalues and eigenfunctions arranged into bands, plot the eigenfunctions corresponding to eigenvalues of multiplicity greater than 1
	INPUTS:
		repInfo: bool array, array of flags that determine whether an eigenvalue has multiplicity greater than 1 - 2nd output of BandsWithMultiplicity
		bands, bandEFs: eigenvalue and eigenfunctions organised into bands via GetBands()
		nN: int, if VP is True, this is the number of gridpoints to use in plotting, if VP is False this is required to be the number of gridpoints that were used in each dimension in the FDM mesh
		nMultPlots: int, number of eigenvalues to create plots for, the order is determined by the ordering of repInfo. If 0, we will plot ALL such eigenvalues!
		saveTo: str, if empty, then we will return a list of all figure handles that we generated. If provided, we will save figures to the directory provided (using appropriate naming conventions) rather than displaying them and returning them in a list.
		saveType: str, file extension for figure plots, default is .pdf.
		VP: bool, if True then we are handling VP information, otherwise we are handling FDM information
	OUTPUTS:
		figList: list of matplotlib.pyplot.figures, each figure contains plots for the eigenfunctions corresponding to eigenvalues of multiplicity greater than 1. Is an empty list if saveTo is provided.
	'''
	
	# differences in storage between VP and FDM
	if VP:
		e = 3
	else:
		e = 2
	# Let's plot (a subset of) the eigenfunctions that correspond to eigenvalues of multiplicity greater than 1
	# there's likely a fancy way to do this, by getting all the eigenvalues of multiplicity greater than 1 extracted first, then plotting etc.
	# But since that's a lot of effort and we don't need this information beyond plotting, a while loop is going to suffice
	
	# tracker variables
	i = 0
	plotsMade = 0
	if saveTo:
		# if a string is provided, then this is the output directory for figure files, and we don't want to return a list
		timeStamp = datetime.today().strftime('%Y-%m-%d-%H-%M')
		# check that saveTo was provided with correct syntax at the end, IE is a directory and not a file
		if saveTo[-1]!='/':
			saveDir = saveTo + '/'
		else:
			saveDir = saveTo
		if saveType[0]!='.':
			pType = '.' + saveType
		else:
			pType = saveType
	# storage for output
	figList = []
		
	# check default behaviour
	if nMultPlots==0:
		# we want them all then, I guess.
		print('Creating plots for all eigenvalues of multiplicity greater than one...')
		nMP = repInfo.shape[0]
	else:
		# limit to number of plots that the user wants
		nMP = nMultPlots
		print('Creating plots for first %d eigenvalues of multiplicity greater than one...' % nMP)
	
	# step through the array repInfo, stopping once we've made as many plots are we're told, or we run out of rows to look for repeats in
	while (i<repInfo.shape[0]) and (plotsMade<nMP):
		# step through the array notRepeats
		# for each row check if a repeat occured, and if so create a plot of the eigenfunctions corresponding to repeated eigenvalues
		if (not np.all(repInfo[i,:])):
			# incriment counter variable!
			plotsMade += 1
			# if there's at least one False in this row, a repeat eigenvalue occurred
			# there could be more than one eigenvalue repeated though, so we need to be careful
			# first, we need to locate each "false" entry in notRepeat[i,:]
			# this eigenvalue is then a repeat of the one "to the left" in the array
			repeatInds = np.logical_not(repInfo[i,:]).nonzero()[0]
			# repeatInds is an array containing the indices of all the "False" entries in this row of notRepeats
			# if consecutive values in repeatInds are consective integers, then there is an eigenvalue of multiplicity 1 + the number of consecutive integers
			# if consecutive values in repeatInds are not consecutive integers, then there are multiple eigenvalues of multiplicity greater than one in this row
			# what we will do is extract each "set" of repeated eigenvalues into a list, then plot using the information in each of these lists
			toPlot = []
			riStartInd = 0
			riEndInd = 0
			while riStartInd<repeatInds.shape[0]:
				# whilst we still have entries in repeatInds to compare to
				if riEndInd==repeatInds.shape[0]-1:
					# we are at the last entry, and cannot look any further. 
					# So the final set of consecutive integers occur from riStartInd to riEndInd now
					toPlot.append([repeatInds[riStartInd], repeatInds[riEndInd]])
					break
				else:
					# we are not at the last entry, so can look forwards to see if repeatInds[riEndInd]+1 = repeatInds[riEndInd+1]
					if repeatInds[riEndInd]+1==repeatInds[riEndInd+1]:
						# these are consecutive integers, so keep looking onwards for repeats
						riEndInd += 1
					else:
						# these entries are not consecutive integers, so there is a set of repeated eigenvalues from riStartInd through to riEndInd (that repeats the eigenvalue at repeatInds[riStartInd]-1)
						# first, append to our tracking list
						toPlot.append([repeatInds[riStartInd], repeatInds[riEndInd]])
						# now update our trackers so we can look for the next set of repeated eigenvalues
						riStartInd = riEndInd + 1
						riEndInd += 1

			# toPlot is now a list of 2-lists, [ [start, end], [start, end], ...] etc where
			# repeatInds[start]-1 through repeatInds[end] correspond to the same eigenvalue, for each 2-list
			# now, we just plot each set of eigenfunctions specified by the 2-lists of toPlot on the same axes...
			for p in toPlot:
				multip = p[1]-p[0]+e
				# create figure. Do not share axis because this does some funky stuff for 2D arrays!
				fig, axs = plt.subplots(nrows=2, ncols=multip, sharex=False, sharey=False)
				# top "row" of axs will hold real part, bottom part will hold imaginary part
				for pp in range(multip):
					# the eigenfunction to plot is FDM_bandFns[p[0]-2+pp-1]][i,2:]
					# -e from QM offset, -1 from how start indicates the eigenvalues that repeat
					if VP:
						# plot using VP methods
						rC, iC = VP_InsertToPlot(axs[:,pp], bandEFs[p[0]-e+pp-1][i,:][e:], nPts=nN)
					else:
						# plot using FDM methods
						rC, iC = FDM_InsertToPlot(axs[:,pp], nN, TranslateFDMef(bandEFs[p[0]-e+pp-1][i,:])[2:])
					fig.colorbar(rC, ax=axs[0,pp])
					fig.colorbar(iC, ax=axs[1,pp])
				# figure title and labelling if necessary
				params = bands[p[0]-e+pp-1][i,:]
				fig.suptitle(r'$\theta = [%.3f, %.3f]\pi, \ \omega = %.5f$ (multiplicity %d)' % (params[0]/pi, params[1]/pi, params[2], multip))
				if saveTo:
					# generate filename and save figure to file format
					bandNo = p[0]-e
					figName = saveDir + timeStamp + '-ev%d-band%d' % (plotsMade, bandNo) + pType 
					fig.savefig(figName, bbox_inches='tight')
					plt.close(fig)
				else:
					# append figure to list
					figList.append(fig)
		# after all is said and done, incriment our stepper
		i += 1
	# having exited the while loop, we can now return the list of figures that we created
	return figList

def TranslateFDMef(u):
	'''
	Translates an eigenfunction from the FDM onto the domain of the VP. 
	The domains are identical, however the period cell was viewed in different ways: FDM has the central vertex at (1/2,1/2), whilst VP had the vertex at (0,0).
	As such, we need to move the entries of the vector u containing the FDM entries, so that we can view the two functions over the same reference domain.
	INPUTS:
		u: (2+(N-1)**2,) complex, eigenfunctions from FDM solve
	OUTPUTS:
		uShift: (2+(N-1)**2,) complex, eigenfunction u but now on the reference domain of VP
	'''
	
	# N from FDM; FDM has shape (2+(N-1)*(N-1),) due to the periodicity constraint
	N = int(np.sqrt( np.shape(u)[0] - 2 ) + 1)
	
	# FDM uses vertex at (1/2,1/2), whereas VP uses vertex at (0,0), so we need to translate by periodicity
	# this is compounded by the "repeated values" on the slave nodes in each of the arrays. So we don't want to re-insert the slave nodes just yet, as we want to translate first
	# for ease, let's move from our vector to the matrix representation
	matFDM = u[2:].reshape(N-1,N-1)

	swaps = np.zeros_like(matFDM)
	swaps[0:N//2,0:N//2] = matFDM[N//2:,N//2:] # top right to bottom left
	swaps[N//2:,N//2:] = matFDM[0:N//2,0:N//2] # bottom right to top left
	swaps[0:N//2,N//2:] = matFDM[N//2:,0:N//2] # bottom right to top left
	swaps[N//2:,0:N//2] = matFDM[0:N//2,N//2:] # top left to bottom right
	
	uShift = u
	uShift[2:] = swaps.reshape(((N-1)*(N-1),)) # replace vector of grid values
	return uShift

#%% Comparing eigenvalues from the two methods

def BandPlot(bVP=[], bFDM=[], commonAxis=True):
	'''
	Creates a (1D) plot of the eigenvalues (colour-coded by band) for the spectral bands from the VP and FDM runs.
	Can place these on a common axis too for comparitive purposes.
	Only supplying one of bVP or bFDM will ignore plotting of the other spectrum
	INPUTS:
		bVP: list, list where each index is a numpy array whose last column index contains the eigenvalues of that band. Output of the VP problem
		bFDM: list, as above but as an output of the FDM problem
		commonAxis: bool, if True and bVP and vFDM are supplied, plots are created on a common axis
	OUTPUTS:
		figList: list, entries are lists of two elements, [fig, ax] for each of the plots requested
	'''

	# number of bands in each list
	nbandsVP = len(bVP)
	nbandsFDM = len(bFDM)
	# this will be the output
	figList = []
	
	if bVP and bFDM and commonAxis:
		# both are supplied and want displayed on a common axis
		fig, ax = plt.subplots(1)
		ax.set_xlabel(r'$\omega$')
		ax.set_title(r'Spectrum comparison: FDM vs VP')
		bandNo = 0
		yticks = []
		yticklabels = []
		# for bands where we have both VP and FDM information, combine these so the colour scheme works
		for b in range(min(nbandsVP,nbandsFDM)):
			specVals = np.hstack((bVP[b][:,-1], bFDM[b][:,-1]))
			# plot FDM vals at y=0, and VP vals at y=1
			yAxVals = np.hstack((np.ones_like(bVP[b][:,-1])+b/(nbandsVP+1), np.zeros_like(bFDM[b][:,-1])+b/(nbandsFDM+1)))
			ax.scatter(specVals, yAxVals, marker='x', s=1, label='Band %d' % (bandNo+1))
			# add axis labels for the bands
			yticks.append(1+b/(nbandsVP+1))
			yticks.append(b/(nbandsFDM+1))
			yticklabels.append(r'VP Band %d' % (bandNo+1))
			yticklabels.append(r'FDM Band %d' % (bandNo+1))
			# move onto the next band
			bandNo += 1
		# only one of these two loops then executes, depending on which of bVP or vFDM had more bands in it
		for b in range(bandNo, nbandsVP):
			# plot remaining vBP bands
			ax.scatter(bVP[b][:,-1], np.ones_like(bVP[b][:,-1])+b/(nbandsVP+1), marker='x', s=2, label='Band %d' % (b))
			yticks.append(1+b/(nbandsVP+1))
			yticklabels.append(r'VP Band %d' % (b+1))
		for b in range(bandNo, nbandsFDM):
			# plot remaining vFDM bands
			ax.scatter(bFDM[b][:,-1], np.zeros_like(bFDM[b][:,-1])+b/(nbandsFDM+1), marker='x', s=2, label='Band %d' % (b))
			yticks.append(b/(nbandsFDM+1))
			yticklabels.append(r'FDM Band %d' % (b+1))
		# add band labels on y axis
		ax.set_yticks(yticks)
		ax.set_yticklabels(yticklabels)
		figList.append([fig, ax])
	else:
		# we don't want a combined plot
		
		if bVP:
			# want to plot bVP
			figVP, axVP = plt.subplots(1)
			axVP.set_xlabel(r'$\omega$')
			axVP.set_title(r'Spectrum (VP)')
			ytickVP = []
			yticklabelVP = []
			# plot all of the bands
			for b in range(nbandsVP):
				axVP.scatter(bVP[b][:,-1], np.zeros_like(bVP[b][:,-1])+b/(nbandsVP+1), marker='x', s=2, label='Band %d' % (b))
				ytickVP.append(b/(nbandsVP+1))
				yticklabelVP.append('Band %d' % b)
			# add band labels
			axVP.set_yticks(ytickVP)
			axVP.set_yticklabels(yticklabelVP)
			figList.append([figVP, axVP])
		if bFDM:
			# want to plot bFDM
			figFDM, axFDM = plt.subplots(1)
			axFDM.set_xlabel(r'$\omega$')
			axFDM.set_title(r'Spectrum (FDM)')
			ytickFDM = []
			yticklabelFDM = []
			# plot all of the bands
			for b in range(nbandsFDM):
				axFDM.scatter(bFDM[b][:,-1], np.zeros_like(bFDM[b][:,-1])+b/(nbandsFDM+1), marker='x', s=2, label='Band %d' % (b))
				ytickFDM.append(b/(nbandsFDM+1))
				yticklabelFDM.append('Band %d' % b)
			# add band labels
			axFDM.set_yticks(ytickFDM)
			axFDM.set_yticklabels(yticklabelFDM)
			figList.append([figFDM, axFDM])	
	# return whatever figures we created
	return figList

def EvalProximity(eVP, eFDM):
	'''
	Computes and analyses how close the eigenvalues (for a given band if necessary) are for the same value of theta.
	INPUTS:
		eVP: (nEvals, 3) float, theta1, theta2, omega values stored row-wise and computed by VP
		eFDM: (nEvals, 3) float, as for eVP but computed via FDM
	Inputs are assumed to have their rows with matching theta1, theta2 values, although this is checked for in the program anyway.
	OUTPUTS:
		
	'''
	
	# first, we can do a reality check on whether the theta values match up
	thetaDiffs = eVP[:,0:2] - eFDM[:,0:2]
	if np.any(np.linalg.norm(thetaDiffs, axis=1) >= 1e-8):
		# if the difference vector of any of the quasimomentum pairs is far from zero
		print('WARNING: Theta values do not appear to match across arrays')
	
	eDiff = (eVP[:,-1] - eFDM[:,-1]).reshape(np.shape(eVP)[0],1)
	print('Differences are eVP - eFDM for reference')
	print('Max diff: %.5e' % np.max(eDiff))
	print('Min diff: %.5f' % np.min(eDiff))
	
	fig, ax = PlotEvals(np.hstack((eVP[:,0:2], eDiff)), 'heat', title='Eigenvalue differences')
	
	return fig, ax

#%% Cross-checking and analysis

def VP_EvalOrderCheck(evs, efs, tol=1e-8):
	'''
	Given the eigenvalues and eigenfunction information from the variational problem solve, check to see whether eigenvalues were found in order.
	Ignore eigenvalues for which there was no convergence.
	INPUTS:
		evs: raw load from data file, output of VP solver
		efs: raw load from eigenfunction data file, output of VP solver
		tol: float, tolerance to which to treat inner products as zero
	OUTPUTS:
		evs_out, efs_out: eVals and eFuncs having been swept for correct ordering and additional convergence concerns
		iList: list of test results to be followed up on
	'''
	
	# outputs: once we're confident this is working and we don't need to debug, just edit the inputs directly
	evs_out = np.copy(evs)
	efs_out = copy(efs)
	iList = []
	# first, check how many eigenvalues we attempted to compute
	nEvals = evs_out.shape[1]-3
	# now the tedious bit, for each row (fixed theta) check whether the eigenvalues are in order
	for row in range(evs_out.shape[0]):
		stayOnRow = True
		nSweeps = 0
		while stayOnRow and nSweeps<nEvals and evs_out[row,0]!=0:
			# continue examining this row until either; we conclude our tests and want to move on, we overstep a maximum number of sweeps over this row, or if none of these eigenvalues converged (so we ignore the row anyway)
			convFlag = evs_out[row,0]
			# only take eigenvalues for which there was convergence
			if convFlag < 0:
				# convFlag = -1 when there were no convergence issues
				eVals = np.copy(evs_out[row,3:])
			else:
				# row 3+convFlag is the first eigenvalue for which there was no convergence
				# note that this array is non-empty as convFlag>0
				eVals = np.copy(evs_out[row,3:3+int(convFlag)])
			# we now check that eVals is in ascending order
			if not np.all(eVals[:-1] <= eVals[1:]):
				# if all entries are in ascending order, move on to the next row (this evaluates to false)
				# otherwise, we have some investigating to do
				print('-----')
				print('Row %d has eigenvalues unsorted, eVals: ' % row, end='')
				print(eVals)
				# determine the eigenvalues that are out of order
				outOfOrder = np.where(eVals[:-1]-eVals[1:] > 0)[0][0]
				# outOfOrder and outOfOrder+1 are in the wrong order. 
				# Note that there might be more than one pair of eigenvalues in the wrong order - our solution to this will be to repeat the loop at this value of row
				# these are the two Poly2D objects that correspond to the eigenfunctions we've found
				p1 = copy(efs_out[row*nEvals + outOfOrder])
				p2 = copy(efs_out[row*nEvals + outOfOrder+1])
				# we can check whether these functions are orthogonal to each other and to the other eigenfunctions that come before them...
				print('Checking orthogonality...', end='')
				if np.linalg.norm(p1.ip(p2))>tol:
					print('\n    Out-of-order eigenfunctions are NOT orthogonal')
				else:
					print('[confirmed]')
					
				# if there is no issue with orthogonality, the next thing to check is whether the numerical scheme simply converged to a higher eigenvalue due to our starting guess and current constraints.
				# as such, we will test the following: does the VP solve converge to p2 if given the previous eigenfunctions EXCEPT p1, and when giving a starting guess of p2?
				# and, does the numerical scheme converge to p1 if given the previous eigenfunctions AND p2, when given a starting guess of p2 (just to speed things up)
				# flags for passing each test
				p2Test = True
				p1Test = True
				convFail = False
				print('Testing VP convergence to p2 given close starting guess:') 
				p2Answer, omegaSq2, noConv2 = VarProb(p2.M, 1, p2.theta, nIts=2500, prevUs=efs[row*nEvals:row*nEvals+outOfOrder], x0=p2.uCoeffs, lOff=True)
				if noConv2 > -0.5:
					# numerical scheme didn't even converge
					print('    NO CONVERGENCE IN NUMERICAL SCHEME')
					p2Test = False
					convFail = True
				elif np.abs(np.sqrt(omegaSq2)-np.sqrt(p2.lbda))<tol:
					# if we start AT p2, we converge to p2. So that's fine, it seems like we've just found them out of order
					print('    Found p2')
					p2Test = True
				elif np.abs(np.sqrt(omegaSq2)-np.sqrt(p1.lbda))<tol:
					# if we start AT p2, we still converge to p1. This is odd, and really shouldn't happen
					print('    Found p1 !!')
					p2Test = False
				else:
					# we didn't converge to either of p1 or p2, weirdly
					print('    CONVERGED TO NEITHER, omega found: %.8f' % np.sqrt(omegaSq2))
					p2Test = False
					# check whether the coefficients are close?
	# 				print('    Max. abs. diff between p2 and converged solution coeffs: %.8f' % (np.max(np.abs(Real2Comp(p2Answer[0,:])-p2.uCoeffs))))
	# 				print('   |omega^2-p2.lbda|: %.5e' % np.abs(omegaSq2 - p2.lbda))
	# 				print('    Max. abs. diff between p1 and converged solution coeffs: %.8f' % (np.max(np.abs(Real2Comp(p2Answer[0,:])-p1.uCoeffs))))
	# 				print('   |omega^2-p1.lbda|: %.5e' % np.abs(omegaSq2 - p1.lbda))
					
				# now check that we find p1 if we start at p1 and pass in p2
				print('Testing VP convergence to p1 given p2 as previous eigenfunction:')
				# construct "previous eigenfunction" list, including p2 on the end
				pList = efs[row*nEvals:row*nEvals+outOfOrder]
				pList.append(p2)
				p1Answer, omegaSq1, noConv1 = VarProb(p1.M, 1, p1.theta, nIts=2500, prevUs=pList, x0=p1.uCoeffs, lOff=True)
				if noConv1 > -0.5:
					# numerical scheme didn't even converge
					print('    NO CONVERGENCE IN NUMERICAL SCHEME')
					p1Test = False
				elif np.abs(np.sqrt(omegaSq1)-np.sqrt(p1.lbda))<tol:
					# if we start AT p1, we converge to p1. So that's fine, it seems like we've just found them out of order
					print('    Found p1')
					p1Test = True
				elif np.abs(np.sqrt(omegaSq1)-np.sqrt(p2.lbda))<tol:
					# if we start AT p1, we still converge to p2. 
					# This shouldn't happen if the orthogonality condition is unbroken
					print('    Found p2 - is orthogonality condition really holding?')
					p1Test = False
				else:
					# we didn't converge to either of p1 or p2, weirdly
					print('    CONVERGED TO NEITHER, omega found: %.8f' % np.sqrt(omegaSq1))
					p1Test = False
					# check whether the coefficients are close?
	# 				print('    Max. abs. diff between p1 and converged solution coeffs: %.8f' % (np.max(np.abs(Real2Comp(p2Answer[0,:])-p1.uCoeffs))))
	# 				print('   |omega^2-p1.lbda|: %.5e' % np.abs(omegaSq1 - p1.lbda))
	# 				print('    Max. abs. diff between p2 and converged solution coeffs: %.8f' % (np.max(np.abs(Real2Comp(p2Answer[0,:])-p2.uCoeffs))))
	# 				print('   |omega^2-p2.lbda|: %.5e' % np.abs(omegaSq1 - p2.lbda))
				
				# the resulting action we take depends on the outcome of the above tests
				# if p2Test & p1Test, then these are both valid eigenfunctions, we just set our starting guess too close to the "higher one". In fact, it's likely that this eigenvalue (in reality) has multiplicity greater than 1, and the numerical approximation isn't precise enough to pick up on this.
				# if p2Test is true and p1Test is false, chances are we missed the "lower" eigenvalue, which is then causing a knock-on effect in computation of the higher-up eigenfunctions.
				# if p2Test is false but p1Test is true, chances are that both eigenfunction approximations are rubbish: we don't find p2 if we start near it w/o orthogonality to p1, but we DO find p1 if we start near p1 and have orthogonality to p2? Flag for further investigation, and mark all eigenvalues beyond this repetition as notCoverged just to be safe
				# if p2Test & p1Test both fail, chances are our approximations have been improved slightly because of a better starting guess.
				if p2Test and p1Test:
					# swap the order of the eigenvalues and the eigenfunctions in the respective arrays/lists
					evs_out[row, 3+outOfOrder] = eVals[outOfOrder+1]
					evs_out[row, 3+outOfOrder+1] = eVals[outOfOrder]
					efs_out[row*nEvals+outOfOrder] = p2
					efs_out[row*nEvals+outOfOrder+1] = p1
					# we should go over this row again to check for other values that we need to resolve
					stayOnRow = True
					print('   Resolved row %d: swapped evals and functions %d <-> %d' % (row, outOfOrder, outOfOrder+1))
				elif p2Test and (not p1Test):
					# we believe the lower eigenvalue is correct, but missing it has introduced the "rubbish" eigenfunction p1, which we don't trust
					# so, swap the eigenvalues and functions around, then flag what was p1 as a non-convergence point - if we want to, we could retry running the loop again from here, but effort
					evs_out[row, 3+outOfOrder] = eVals[outOfOrder+1]
					evs_out[row, 3+outOfOrder+1] = eVals[outOfOrder]
					efs_out[row*nEvals+outOfOrder] = p2
					efs_out[row*nEvals+outOfOrder+1] = p1
					# flag no convergence
					evs_out[row, 0] = outOfOrder+1
					# move onto next row, no need to resweep
					stayOnRow = False
					print('   Resolved row %d: swapped evals and functions %d <-> %d, set noConv to %d' % (row, outOfOrder, outOfOrder+1, outOfOrder+1))
				elif (not p2Test) and p1Test:
					# we don't find p2 if we start near it w/o orthogonality to p1, but we DO find p1 if we start near p1 and have orthogonality to p2? Flag for further investigation, and mark all eigenvalues beyond this repetition as notCoverged just to be safe
					evs_out[row, 0] = outOfOrder
					print('   Flag row %d, evals %d and %d for follow up [test]' % (row, outOfOrder, outOfOrder+1))
					# move onto next row, no need to resweep
					stayOnRow = False
				elif convFail:
					# if one of the retry convergences failed, and additionally both p1Test and p2Test are both false, trust nothing again.
					evs_out[row, 0] = outOfOrder
					print('   Flag row %d, evals %d and %d for follow up [no numerical scheme convergence]' % (row, outOfOrder, outOfOrder+1))
					# move onto next row, no need to resweep
					stayOnRow = False
				else:
					# here we have a lot of questions; foremost is whether when conducting p2Test we found an eigenfunction that corresponds to an eigenvalue LOWER than either of the two we already had
					# in this case, just flag everything as not converging, but append to our "info list" our report so we can investigate further
					iList.append([row, outOfOrder, p1Answer, p2Answer])
					evs_out[row, 0] = outOfOrder
					# resweep in case this hasn't fixed the issue
					stayOnRow = False
					print('   Resolved row %d: flagged for follow up [found new EF]' % row)
			# for debugging, can manually escape loop here
			nSweeps += 1
			if nSweeps>nEvals:
				stayOnRow = False
				print('While loop catch triggered')
	return evs_out, efs_out, iList

#%% Extraction and data handling

def GetBands(evInfo, efInfo=None, VP=True, tol=1e-8):
	'''
	Split the data from either a VP or FDM run into a list of band information.
	Handle the eigenfunctions if they are passed too.
	INPUTS:
		evInfo: raw load from data file, output of either VP or FDM solvers
		efInfo: raw load from eigenfunction data file, output of either VP or FDM solvers
		VP: bool, if True, we are handling VP data. Otherwise, we're handling FDM
		tol: float, passed to GetBands(), error threshold
	OUTPUTS:
		bands: list, each index contains the eigenvalues of the corresponding band
		efBands: list, each index contains the eigenfunctions of the corresponding band
	'''
	
	# initialise lists
	bands = []
	# handle eigenfunctions too?
	if efInfo is None:
		efs = False
		efBands = None
	else:
		efs = True
		efBands = []
		
	# nEvals is computed slightly differently for VP and FDM
	if VP:
		nEvals = np.shape(evInfo)[1] - 3
	else:
		nEvals = np.shape(evInfo)[1] - 2
		
	# get each of the bands in sequence
	if VP:
		for n in range(nEvals):
			bands.append(VP_GetBand(n+1, evInfo))
			if efs:
				efBands.append(efInfo[n::nEvals])
	else:
		for n in range(nEvals):
			bands.append(FDM_GetBand(n+1, evInfo, tol=tol))
			if efs:
				efBands.append(efInfo[n::nEvals,:])
	# return extracted info
	return bands, efBands

# Determine whether we have found any eigenvalues of multiplicity greater than 1
def RepeatedEvals(evInfo, VP=False, rtol=1e-5, atol=1e-8):
	'''
	Given the eigenvalue information, return a vector of tf values where evInfo[tf,:] contain QM values which appear to have repeated eigenvalues (of e-vals of multiplicity greater than 1).
	INPUTS:
		evInfo: eigenvalue information, output of FDM_Load or VP_Load
		VP: bool, if True then we are dealing with VP outputs, otherwise handing FDM outputs
		rtol, atol: see numpy.isclose()
	OUTPUTS:
		tf: bool array, evInfo[tf,:] returns the rows in which there are repeated eigenvalues
	'''
	
	# depending on whether this is VP or FDM data, the QM values start at different column indices
	if VP:
		e = 3
	else:
		e = 2
	
	nEvals = np.shape(evInfo)[1] - e
	# assume that no rows have repeated eigenvalues at the start
	tf = np.zeros((np.shape(evInfo)[0],), dtype=bool)
	
	# compare each of the eigenvalues computed, for each QM, to the others.
	# Return the tf array of values that correspond to the rows in which these repeats occur
	for n in range(e, e+nEvals-1):
		for m in range(n+1, e+nEvals):
			# find eigenvalues in this column that are near to the eigenvalues in the next column
			match = np.isclose(evInfo[:,n], evInfo[:,m], rtol=rtol, atol=atol)
			print('Found %d repeated eigenvalues between band %d and %d' % (np.sum(match), n-e+1, m-e+1))
			# if there were any pairs that were close, flag that row in tf
			tf = tf | match
	
	print('Found a total of %d repeated eigenvalues' % np.sum(tf))
	return tf

# Create a list of the spectral bands: eigenvalues of multiplcity one are grouped into a single band
def BandsWithMultiplicity(ev, VP=True, rtol=1e-5, atol=1e-8):
	'''
	Returns a list of the spectral bands, but accounting for possible multiplicity in the eigenvalues that we have computed.
	INPUTS:
		ev: eigenvalue information, output of loading the eigenvalues from files
		VP: bool, if True, input is from VP, otherwise assumed from FDM
		rtol, atol: see np.isclose().
	OUTPUTS:
		bands: list of numpy arrays, bands[i][j,:] = [theta0, theta1, omega_j]
		notRepeat: bool, array of flags that determine whether an eigenvalue has multiplicity greater than 1
	'''

	# how many eigenvalues did we try to compute?
	if VP:
		e = 3
	else:
		e = 2
	nEvals = ev.shape[1] - e
	print('Found at most %d eigenvalues per QM' % nEvals)
	
	# create overestimates for the arrays that will hold the eigenvalues
	# fill with -1s to indicate "bad" eigenvalues, which we should discard upon our cleanup
	bandArray = -1. * np.ones_like(ev)
	# this array tracks which indices in each band we should be using
	notRepeat = np.ones_like(ev, dtype=bool)
	
	for n in range(e,ev.shape[1]-1):
		for m in range(n+1,ev.shape[1]):
			# check whether there are any eigenvalues in close proximity between bands n and m, for this QM
			matches = np.isclose(ev[:,n], ev[:,m])
			# matches is True where eigenvalues are repeated between bands n and m
			# flag any eigenvalues that are repeated in notRepeat
			notRepeat[:,m] = notRepeat[:,m] & (np.logical_not(matches))
			print('Found %d repeats between bands %d and %d' % (matches.sum(), n-e+1, m-e+1))
	
	# notRepeat now contains False wherever an eigenvalue was repeated.
	# We now need to "compress" the values into the correct bands
	nActualBands = np.sum(notRepeat, axis=1)
	# nActualBands[i] = number of distinct eigenvalues ev[i,:] contains
	for i in range(ev.shape[0]):
		# the eigenvalues to insert into bandArray are flagged by notRepeat[i,:]
		# and there are nActalBands[i] to copy in
		# we can also copy in the QM values too
		bandArray[i,:nActualBands[i]] = ev[i,notRepeat[i,:]]
		
	# bandArray now contains the bands with repeat eigenvalues removed, 
	# with -1. entries where these eigenvalues would have been
	# we can now get the bands fairly easily
	bands_full, _ = GetBands(bandArray, None, VP, atol)
	
	# now remove any eigenvalues that are equal to -1 from these bands
	# these are entries in bandArray that have never been set, so are leftover space in our overestimation of how many distinct eigenvalues we had
	bands = []
	for b in bands_full:
		bands.append( b[ b[:,2] > -0.5, : ] )
		
	# and finally, bands is now the list of bands with eigenvalues of multiplicity greater than one accounted for
	return bands, notRepeat

#%% Command-line proceedures

if __name__=="__main__":
	
	# proceedures to carry out
	compareSpectrums = True # plot spectral bands from each method on same axis, and create a figure for it
	multFilter = False # account for multiplicity of eigenvalues before carrying out plotting
	omegaQMPlots = False # plot omega as a function of theta 1 and theta 2 for each method
	nonSimplePlots = False # plot eigenfunctions corresponding to eigenvalues of multiplicity greater than 1
	
	# file dumps to append
	VP_fd = './CompMes_VP_Results/'
	FDM_fd = './FDM_Results/'
	# location to save generated figures to
	fdm_save = FDM_fd + 'N71-alpha0/'
	vp_save = VP_fd
	# for figures that involve data from both, default output directory will be current working directory
	saveDir = './'
	timeStamp = datetime.today().strftime('%Y-%m-%d-%H-%M')
	print('Outputs will be placed in the following directories:')
	print(' [FDM] %s' % fdm_save)
	print(' [VP] %s' % vp_save)
	print(' [Combined] %s' % saveDir)
	
	# information about eigenvalues
	#FDM_searchPath = FDM_fd + 'nPts51-N71-alpha1.csv'
	FDM_searchPath = FDM_fd + 'nPts51-N251-10evals.csv'
	#FDM_searchPath = FDM_fd + 'nPts51-N71.csv'
	FDM_ev, FDM_ef = FDM_Load(FDM_searchPath, funsToo=False)
	N = 71 #this is the number of gridpoints we used
	
	VP_searchPath = VP_fd + 'nPts25-local-M12-N5-t1loops0-24.csv'
	#VP_searchPath = VP_fd + 'nPts25-M10-N2-t1loops0-24.csv'
	#VP_searchPath = VP_fd + 'nPts25-nEvals1.csv'
	VP_ev, VP_ef = VP_Load(VP_searchPath)
	# here, we check for out-of-order eigenvalues
	#VP_ev_out, VP_ef_out, VP_iList = VP_EvalOrderCheck(VP_ev, VP_ef)
	# this is the number of bands we tried to compute
	FDM_nEvals = np.shape(FDM_ev)[1] - 2
	VP_nEvals = np.shape(VP_ev)[1] - 3

	#VP_EvalOrderCheck threw up some results demonstrating that without global optimisation, we can miss minimisers

#%%
	# get all the bands, since we don't expect to have many, just use a list
	FDM_bands, FDM_bandFns = GetBands(FDM_ev, FDM_ef, False)
	VP_bands, VP_bandFns = GetBands(VP_ev, VP_ef, True)
	# now get the bands but filtering out the repeated eigenvalues
	print('--- Filtering FDM for multiplicities...')
	FDM_bandsNM, FDM_NotRep = BandsWithMultiplicity(FDM_ev, False)
	print('--- Filtering VP for multiplicities...')
	VP_bandsNM, VP_NotRep = BandsWithMultiplicity(VP_ev, True)

	# make a plot that compares the bands we obtained from the two methods, using the data we imported
	if compareSpectrums:
		print('Creating spectral band comparison plot')
		figName = saveDir + timeStamp + '-bandPlot.pdf'
		if multFilter:
			fig_SpecComp = BandPlot(VP_bandsNM, FDM_bandsNM)
		else:
			fig_SpecComp = BandPlot(VP_bands, FDM_bands)
		fig_SpecComp[0][0].savefig(figName, bbox_inches='tight')
	
	# create individual plots for how the eigenvalue found depends on the QM
	if omegaQMPlots:
		nVP_bands = 0
		nFDM_bands = 10
		availableTypes = ['scatter','surf','contour','heat']
		pType = availableTypes[3]
		# don't request more bands that you have information for
		if nVP_bands>len(VP_bands) or nVP_bands>len(VP_bandsNM):
			raise ValueError('You don\'t have enough (VP) bands to plot me!')
		elif nFDM_bands>len(FDM_bands) or nFDM_bands>len(FDM_bandsNM):
			raise ValueError('You don\'t have enough (FDM) bands to plot me!')
		# plot bands, starting from band 0 for each
		if multFilter:
			print('Creating omega(theta) plots for VP')
			for b in range(nVP_bands):
				vpTitle = r'(VP) Band %d' % (b+1)
				fObjs = PlotEvals(VP_bandsNM[b], pType, title=vpTitle)
				figNameVP = vp_save + timeStamp + '-VPband' + str(b+1) + '.pdf'
				fObjs[0].savefig(figNameVP, bbox_inches='tight')
			print('Creating omega(theta) plots for FDM')
			for b in range(nFDM_bands):
				fdmTitle = r'(FDM) Band %d' % (b+1)
				fObjs = PlotEvals(FDM_bandsNM[b], pType, title=fdmTitle)
				figNameFDM = fdm_save + timeStamp + '-FDMband' + str(b+1) + '.pdf'
				fObjs[0].savefig(figNameFDM, bbox_inches='tight')
		else:
			print('Creating omega(theta) plots for VP')
			for b in range(nVP_bands):
				vpTitle = r'(VP) Band %d' % (b+1)
				fObjs = PlotEvals(VP_bands[b], pType, title=vpTitle)
				figNameVP = vp_save + timeStamp + '-VPband' + str(b+1) + '.pdf'
				fObjs[0].savefig(figNameVP, bbox_inches='tight')
			print('Creating omega(theta) plots for FDM')
			for b in range(nFDM_bands):
				fdmTitle = r'(FDM) Band %d' % (b+1)
				fObjs = PlotEvals(FDM_bands[b], pType, title=fdmTitle)
				figNameFDM = fdm_save + timeStamp + '-FDMband' + str(b+1) + '.pdf'
				fObjs[0].savefig(figNameFDM, bbox_inches='tight')
	
	# create plots of the eigenfunctions that correspond to non-simple eigenvalues
	if nonSimplePlots:
		# we've already found the eigenvalues. Note that if we didn't pull out the eigenfunctions when loading the FDM data, we'll get an error, so flag it here
		if (not FDM_ef):
			# empty list (unloaded) returns False, so if we get to here we didn't load the data!
			raise ValueError('FDM eigenfunction data not loaded')
		# we're not opening all these plots at once, save them to the output directories above
		# first, for FDM...
		print('Creating plots for non-simple FDM eigenfunctions')
		PlotNonSimpleEVs(FDM_NotRep, FDM_bands, FDM_bandFns, N, nMultPlots=0, VP=False, saveTo=fdm_save)
		# now for VP...
		print('Creating plots for non-simple VP eigenfunctions')
		PlotNonSimpleEVs(VP_NotRep, VP_bands, VP_bandFns, N, nMultPlots=0, VP=True, saveTo=vp_save)
