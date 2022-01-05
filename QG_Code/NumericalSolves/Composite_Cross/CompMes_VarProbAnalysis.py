#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 09:22:05 2021

@author: will

Python file and script containing methods to produce eigenvalue plots from the outputs of:
	CompMesProb_EvalFinder.py [variational problem solver]
"""

import glob

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

def LoadAllFromKey(searchPath):
	'''
	Loads all eigenvalue information from the given search-path into a single eigenvalue array
	INPUTS:
		searchPath: str, string to expand and load eigenvalue information from
	OUTPUTS:
		allEvals: (M,N+3) float, eigenvalue information arrays
	'''
	
	allFiles = glob.glob(searchPath, recursive=False)
	evList = []
	fList = []
	for fname in allFiles:
		e, F = ReadEvals_VarProb(fname, funsToo=True)
		# record e'val array
		evList.append(e)
		# extend list of Poly2D functions
		fList.extend(F)
	# combine e'value arrays
	allEvals = AppendEvalRuns(evList)
	return allEvals, fList

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

#%% Command-line execution

if __name__=='__main__':
	
	fDump = './CompMes_VP_Results/'
	searchPath = fDump + 'nPts25-nEvals1.csv'
	allEvals = LoadAllFromKey(searchPath)
	
	# this is the number of bands we tried to compute
	N = np.shape(allEvals)[1] - 3
	# get all the bands, since we don't expect to have many, just use a list
	bands = []
	for n in range(N):
		bands.append(GetBand(n+1, allEvals, removeFailed=True))
	
	for bi, b in enumerate(bands):
		f, a = PlotEvals(b, pType='heat', title=r'$\omega$ values in band %d' % (bi+1))
		f.show()
	
	# NOTABLE OBSERVATIONS:
		# QM symmetry points look like they're giving the extremities of the spectral bands
		# don't yet know if the bands will overlap
		# low point at qm=0, high point at qm=(+/pi,+/pi) for any combination
		# symmetry in omega wrt qm components is displayed, which is expected