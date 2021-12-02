#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 16:41:37 2021

@author: will

This script handles comparisions between the outputs of CompMes_FDMSolve and CompMesVarProbSolve, including plotting of solutions and graphical representations of the eigenvalues (according to each method)
"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

from CompMes_VarProbAnalysis import LoadAllFromKey as VP_Load
from CompMes_VarProbAnalysis import PlotEvals
from CompMes_VarProbAnalysis import GetBand as VP_GetBand

from CompMes_FDMAnalysis import LoadAllFromKey as FDM_Load
from CompMes_FDMAnalysis import GetBand as FDM_GetBand
from CompMes_FDM import PlotFn as FDM_PlotU
from CompMes_FDM import InsertSlaveNodes

#%% Compare eigenfunctions from the two methods

def CompareEFs(uVP, uFDM, methods=['ratio']):
	'''
	Perform comparisions of the two functions uVP and uFDM, using the methods specified in methods.
	INPUTS:
		uVP: Poly2D, eigenfunction from the VP solve
		uFDM: (2+(N-1)**2,) complex, eigenfunction from the FDM solve
		methods: list of str, defines the comparisions to run on the two eigenfunctions. Currently supported methods are listed below.
	OUTPUTS:
		methResult: dict, keys are the entries of methods, and values are the results of the corresponding comparision.
	
	methods:
		'Linfty': L-infinity norm of the difference uVP - uFDM
	'''

	# initalise results dictionary
	methResult = {}
	# some constants that will be useful throughout
	# N from FDM; FDM has shape (2+(N-1)*(N-1),) due to the periodicity constraint
	N = int(np.sqrt( np.shape(uFDM)[0] - 2 ) + 1)
	# FDM_Gridpoints
	x = y = np.linspace(0,1,endpoint=True,num=N)
	
	# first, load our data into a nicer form for us to work with throughout
	# we will need to evaluate uVP at all pairs of points x,y - thankfully we can do this via meshgrid because of vectorisation
	xx, yy = np.meshgrid(x,y)
	zVP = uVP.val(xx, yy) # zVP[j,i] = uVP.val(x[i],y[j]) due to meshgrid conventions
	# uFDM on the other hand simply contains the values at these points, once we reinsert the periodic boundary and adjust for the shift in domains
	uFDM_Shift = TranslateFDMef(uFDM)
	# now reinsert the slave nodes.
	# note that we need to transpose so that zFDM[j,i] and zVP[j,i] both correspond to the function value at x[i],y[j].
	zFDM = InsertSlaveNodes(N, uFDM_Shift[2:], mat=True).T
	# the two matricies zVP and zFDM now contain corresponding function values element-wise, but the final hurdle is that zFDM has two-norm 1, whilst zVP has our l2-compMes norm 1
	# so we need ot figure out how to compare the two functions given this additional complexity
	
	# compare ratio of function values, see if one is just a rescaling of the other
	# return a list of three elements;
	# index 0 being a boolian array indicating where division by zero didn't take place,
	# index 1 being the result of element-wise division, ignoring places where division by 0 would happen
	# index 2 is a bool which is True if zeros of both zFDM and zVP occurred in the same places
	if 'ratio' in methods:
		# check where zeros of both arrays occur (to a tolerance)
		zerosVP = np.abs(zVP)<=1e-8
		zerosFDM = np.abs(zFDM)<=1e-8
		# if zeros occur in the same places, then we don't need to worry about not being able to divide one function by the other element-wise and getting NaNs
		if np.all(zerosVP & zerosFDM):
			# if zeros occur, they are in the same places
			print('All zero-entries are in the same array locations')
			safeDiv = True
		else:
			# zeros do not occur in the same places
			print('WARNING: zero entries do not occur in same array locations')
			safeDiv = False
		# divide everywhere where it is safe to do so
		zRatio = np.divide(zVP, zFDM, out=np.zeros_like(zVP), where=np.abs(zFDM)!=0.)
		# write dictionary output
		methResult['ratio'] = [np.abs(zFDM)>1e-8, zRatio, safeDiv]
	return methResult

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
			# plot FDM vals at y=1, and VP vals at y=0
			yAxVals = np.hstack((np.zeros_like(bVP[b][:,-1])+b/(nbandsVP+1), np.ones_like(bFDM[b][:,-1])+b/(nbandsFDM+1)))
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
			ax.scatter(bVP[b][:,-1], np.zeros_like(bVP[b][:,-1])+b/(nbandsVP+1), marker='x', s=2, label='Band %d' % (b))
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
#%% Command-line proceedures

if __name__=="__main__":
	
	# file dumps to append
	VP_fd = './CompMes_VP_Results/'
	FDM_fd = './FDM_Results/'
	
	# information about eigenvalues
	FDM_searchPath = FDM_fd + 'nPts25-N31-t1loops0-24.csv'
	FDM_ev, FDM_ef = FDM_Load(FDM_searchPath, funsToo=True)
	
	VP_searchPath = VP_fd + 'nPts25-nEvals1.csv'
	VP_ev, VP_ef = VP_Load(VP_searchPath)
	
	# this is the number of bands we tried to compute
	FDM_nEvals = np.shape(FDM_ev)[1] - 2
	VP_nEvals = np.shape(VP_ev)[1] - 3
	# get all the bands, since we don't expect to have many, just use a list
	FDM_bands = []
	FDM_bandFns = []
	VP_bands = []
	VP_bandFns = []
	for n in range(FDM_nEvals):
		FDM_bands.append(FDM_GetBand(n+1, FDM_ev))
		FDM_bandFns.append(FDM_ef[n::FDM_nEvals,:])
	for n in range(VP_nEvals):
		VP_bands.append(VP_GetBand(n+1, VP_ev))
		VP_bandFns.append(VP_ef[n::VP_nEvals])
		
# 	uFDM = FDM_bandFns[0][83,:]
# 	uVP = VP_bandFns[0][83]
# 	print(uVP)
# 	FDM_PlotU(31, TranslateFDMef(uFDM)[2:])
# 	mr = CompareEFs(uVP, uFDM)
	
	fL = BandPlot(VP_bands, FDM_bands)
	fig, ax = EvalProximity(VP_bands[0], FDM_bands[0])
		
# 	for bi, b in enumerate(FDM_bands):
# 		f, a = PlotEvals(b, pType='heat', title=r'(FDM) $\omega$ values in band %d' % (bi+1))
# 		f.show()
# 		
# 	for bi, b in enumerate(VP_bands):
# 		f, a = PlotEvals(b, pType='heat', title=r'(VP) $\omega$ values in band %d' % (bi+1))
# 		f.show()
