#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 10:44:59 2021

@author: will
"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc, cm
rc('text', usetex=True)
from mpl_toolkits.mplot3d import Axes3D

# for saving data
from datetime import datetime

# for loading files
import glob

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

class EvalRun:
	'''
	Class to hold run data - each instance includes:
		N: int, number of meshpoints in each dimension
		qm: (2,) float, value of the quasi-momentum for the run
		eVals: (n,) complex, computed eigenvalues from the run
		rEvals: (m,) real, real eigenvalues extracted from the computed eigenvalues
	'''
	def __init__(self, N, qm, eVals):
		self.N = N
		self.qm = qm
		self.eVals = eVals
		self.rEvals = np.real(eVals[RealEvalIndices(eVals)])
		
	def eValDensity(self, x):
		'''
		Computes the eigenvalue density function for this run.
		For a value x, returns the ratio (#eVals<x)/#eVals.
		'''
		
		lessThanX = np.sum( self.rEvals <= x )
		totEvals = self.rEvals.shape[0]
		return lessThanX/totEvals
	
# read names of eigenvalue files
eValDump = './EvalDump/'
eValFiles = glob.glob(eValDump + 'EvalsN*.npz')

# for each file, load the data
extractedInfo = []
for file in eValFiles:
	tmp = np.load(file)
	extractedInfo.append(EvalRun(int(tmp['N']), tmp['qm'], tmp['eVals']))
	
# the list extractedInfo now contains all the run data as EvalRun objects
# let's try plotting the eigenvalues on an axes to see what happens as we add more meshpoints...?
targetQM = np.zeros((2,), dtype=float)
fig, ax = plt.subplots(1)
ax.set_xlabel(r'$\omega^2$')
ax.set_ylabel(r'$N$')
ax.set_title(r'Computed eigenvalues')
for run in extractedInfo:
	# check that QM values match the one we want to look at
	if np.allclose(run.qm, targetQM):
		ax.scatter(run.rEvals, run.N * np.ones_like(run.rEvals), s=1, marker='x', label=str(run.N))
ax.legend()
fig.show()

# why not try to plot the eigenvalue densities instead?
for run in extractedInfo:
	# check that QM values match the one we want to look at
	if np.allclose(run.qm, targetQM):
		fig, ax = plt.subplots(1)
		ax.set_xlabel(r'$\omega^2$')
		ax.set_ylabel(r'E-val density')
		ax.set_title(r'$N=' + str(run.N) + r'$')
		x = np.linspace(0, np.max(run.rEvals), num = 2.*run.N*run.N)
		y = np.zeros_like(x)
		for i,xi in enumerate(x):
			y[i] = run.eValDensity(xi)
		ax.plot(x,y)
	fig.show()