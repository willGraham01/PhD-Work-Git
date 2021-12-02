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

def GetBand(band, evs):
	'''
	Extracts the band-th eigenvalue from each row in evs.
	Converts to real numbers, and throws an error up if there is an eigenvalue which cannot be cast to a real number.
	INPUTS:
		band: int, band number to extract eigenvalues of (starting from band 1)
		evs: (nRuns,2+n) float, rows corresponding to nRun eigenvalue runs in which n eigenvalues were computed
	OUTPUTS:
		bandInfo: (nRuns,3) float, columns 0,1 are the quasimomentum values corresponding to the eigenvalue at column 2.
	'''
	
	# this is the number of eigenvalues per QM that was computed
	nEvals = np.shape(evs)[1] - 2
	# check that we have information on this band
	if band>nEvals:
		raise ValueError('Want band %d but only have information up to band %d' % (band, nEvals))
		
	# check for imaginary eigenvalues
	tf = RealEvalIndices(evs[:,1+band])
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
	fList = []
	for fname in allFiles:
		e, F = ReadEvals_FDM(fname, funsToo=funsToo)
		# record e'val array
		evList.append(e)
		# extend list of vector arrays
		fList.append(F)
	# combine e'value arrays
	allEvals = AppendEvalRuns(evList)
	# only return vectors if they were asked for
	if funsToo:
		allVecs = AppendEvalRuns(fList) #note that this works as intended for arrays!
	else:
		allVecs = None
	return allEvals, allVecs

#%% Reality checker functions

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
			dataDisp = ax.plot_trisurf(evs[:,0]/pi, evs[:,1]/pi, evs[:,2], cmap=plt.cm.viridis)
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
	
	fDump = './FDM_Results/'
	searchPath = fDump + 'nPts25-N31-t1loops0-24.csv'#'nPts25-N31.csv'#
	allEvals, allEvecs = LoadAllFromKey(searchPath, funsToo=True)
	
	# this is the number of bands we tried to compute
	nEvals = np.shape(allEvals)[1] - 2
	# get all the bands, since we don't expect to have many, just use a list
	bands = []
	for n in range(nEvals):
		bands.append(GetBand(n+1, allEvals))
	
	for bi, b in enumerate(bands):
		f, a = PlotEvals(b, pType='heat', title=r'$\omega$ values in band %d' % (bi+1))
		f.show()
	
	# NOTABLE OBSERVATIONS:
		# QM symmetry points look like they're giving the extremities of the spectral bands
		# don't yet know if the bands will overlap
		# low point at qm=0, high point at qm=(+/pi,+/pi) for any combination
		# symmetry in omega wrt qm components is displayed, which is expected