#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 13:22:37 2021

@author: will
"""
# sys.argv[1:] contains command line arguments. Index 0 is just the script name
import argparse
# for returning values to the command line
import sys

import numpy as np
from numpy import pi

from CompMeas_VarProb import SolveVarProb

#%% Wrapper for main-line function

def EvalThetaLoop(nPts, fName, M=15, N=3, nIts=5000, lOff=False, saveFuncs=False):
	'''
	Loops over the quasi-momentum values in sequence, writing the eigenvalues to an output file.
	Otuput is .csv with noConv, theta/pi, omega_1,..., omega_N on each row.
	INPUTS:
		nPts: int, number of gridpoints to use in each direction of theta
		fName: str, file or path to file for eigenvalues to be stored in
		M: int, M-1 is the highest order polynomial term used in the approximation of the eigenfunctions
		N: int, the number of eigenvalues to get for each value of theta
		nIts: int, max number of iterations for the minimiser
		lOff: bool, if True then the log will NOT print to the screen
		saveFuncs: bool, if True then a second file will be created containing the eigenfunction coefficients
	OUTPUTS:
		< Writes to file in addition to returning the below >
		badList: list of (2,) float, those theta values at which we did not get convergence
	'''

	unifRange = np.linspace(-1., 1., endpoint=True, num=nPts)
	# Track errors in convergence
	badList = []
	
	# The plan is to save to a .csv file the following information;
	# noConv, theta1/pi, theta2/pi, omega_1, omega_2, ..., omega_N
	# This way, if we terminate our loop early, 
	# we still have some information about the attempts that suceeded
	if saveFuncs:
		# if we also want to save the eigenfunctions, we need to record the
		# value of theta and noConv again, then list the coefficients.
		# Also, make a csv file to contain this data based off fName
		funcFilename = 'funcs_' + fName
	
	for i, t1 in enumerate(unifRange):
		for j, t2 in enumerate(unifRange):
			# Inform about progress if requested
			if (not lOff):
				print( ' << Beginning: (%d,%d) of (%d,%d) >>' % (i,j,nPts-1,nPts-1))
			# Generate QM vector
			theta = np.array([t1, t2], dtype=float) * pi
			uReal, omegaSq, noConv = SolveVarProb(M, N, theta, nIts=nIts, lOff=lOff)
			
			if noConv:
				# if we didn't converge, we should record this value of theta for later
				badList.append(theta)
				
			# write the output, including whether or not we converged
			writeOut = np.hstack((noConv, np.array([t1, t2], dtype=float), np.sqrt(omegaSq)))
			# write the data to the output
			with open(fName, 'ab') as f:
				np.savetxt(f, [writeOut], delimiter=',')
				
			# write the eigenfunctions out if we also wanted them
			if saveFuncs:
				for n in range(N):
					funcWriteOut = np.hstack((noConv, np.array([t1, t2]), uReal[n,:]))
					with open(funcFilename, 'ab') as f:
						np.savetxt(f, [funcWriteOut], delimiter=',')
				
	# return everything that went wrong
	return badList
#%% Command-line execution

if __name__=='__main__':

	parser = argparse.ArgumentParser(description='Approximation of eigenvalues and eigenfunctions for the Dirichlet-to-Neumann map, for the Cross-in-plane geometry.')
	parser.add_argument('-fn', default='', type=str, help='Output file name, if blank filename will be auto-generated')
	parser.add_argument('-fd', default='./CompMesVarProb_Results/', type=str, help='Directory to save output file to.')
	parser.add_argument('-nPts', default=10, type=int, help='Number of gridpoints in each dimension of the quasi-momentum to use.')
	parser.add_argument('-M', default=15, type=int, help='M-1 is the highest order term of the polynomial approximation.')
	parser.add_argument('-N', default=2, type=int, help='Number of eigenfunctions and eigenvalues to find (starting from lowest eigenvalue)')
	parser.add_argument('-nIts', default=5000, type=int, help='Maximum number of iterations for solver')
	parser.add_argument('-lOff', action='store_true', help='Suppress printing of progress and solver log to the screen')
	parser.add_argument('-funcs', action='store_true', help='Write out the eigenfunctions that are found, as well as the eigenfunctions')

	# extract input arguments and get the setup ready
	args = parser.parse_args()	
	
	# Output file handles
	if (not args.fn):
		# if no filename was given, auto-generate
		resultsFile = args.fd + 'VarProbEvals-nPts-%d.csv' % (args.nPts)
	else:
		resultsFile = args.fd + args.fn
	
	# Perform loop at these variable values
	badList = EvalThetaLoop(args.nPts, resultsFile, M=args.M, N=args.N, nIts=args.nIts, lOff=args.lOff, saveFuncs=args.funcs)
	
	print('Did not converge for %d values of theta' % (len(badList)))
	for t in badList:
		print(t)
	if badList:
		# at least one run failed, exit with fail (1)
		sys.exit(1)
	else:
		# all runs converged, exit with sucess (0)
		sys.exit(0)