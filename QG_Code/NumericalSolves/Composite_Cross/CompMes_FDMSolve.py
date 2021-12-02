#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 10:33:05 2021

@author: will

Script is designed to loop over multiple values of theta and compute eigenvalues (and eigenvectors) for the Cross-in-plane geometry via the finite-difference method.
"""

# sys.argv[1:] contains command line arguments. Index 0 is just the script name
import argparse
# for returning values to the command line
import sys

import numpy as np
from numpy import pi

from CompMes_FDM import FDM_FindEvals

#%% Wrapper for main-line function

def FDM_ThetaLoop(nPts, fName, N=31, nEvals=3, alpha3=0., lOff=False, saveFuncs=False, dimSpecs=[False, 0, 0]):
	'''
	Loops over the quasi-momentum values in sequence, writing the eigenvalues to an output file.
	Otuput is .csv with noConv, theta/pi, omega_1,..., omega_N on each row.
	INPUTS:
		nPts: int, number of gridpoints to use in each direction of theta
		fName: str, file or path to file for eigenvalues to be stored in
		N: int, number of spatial gridpoints in each dimension for FDM mesh
		nEvals: int, the number of eigenvalues to get for each value of theta
		nIts: int, max number of iterations for the minimiser
		lOff: bool, if True then the log will NOT print to the screen
		saveFuncs: bool, if True then a second file will be created containing the eigenvector components
		dimSpecs: list(bool, int, int), if 1st entry is True, we only loop over the dimension of theta specificed by the 2nd entry. In this case, the 3rd entry sets the value for the fixed QM component as np.linspace(-1,1,nPts)[tFixed]*pi.
	OUTPUTS:
		< Writes to file in addition to returning the below >
		badList: list of (2,) float, those theta values at which we did not get convergence
	'''

	unifRange = np.linspace(-1,1,num=nPts)
	if saveFuncs:
		# if we also want to save the eigenfunctions, we need to record the
		# value of theta and noConv again, then list the coefficients.
		# Also, make a csv file to contain this data based off fName
		try:
			# try finding where in the string the file extension begins,
			# and inserting funcs there to generate a filename for the 
			extensionIndex = fName.rfind('.')
			funcFilename = fName[:extensionIndex] + '-funcs' + fName[extensionIndex:]
		except:
			# if you couldn't find where the file extension is,
			# just append to the end of the string
			funcFilename = fName + '-funcs'
	
	if dimSpecs[0]:
		# only loop in one dimension
		tFixed = unifRange[dimSpecs[2]]
		if dimSpecs[1]==0:
			# looping over theta1, theta2 = tFixed
			print('1D loop over theta_1, theta_2 fixed at %.3f' % (tFixed*pi))
			for i, t1 in enumerate(unifRange):
				# Inform about progress if requested
				if (not lOff):
					print( ' << Beginning: (%d) of (%d) >>' % (i,nPts-1))
				# Generate QM vector
				theta = np.array([t1, tFixed], dtype=float) * pi
				# compute eigenvalues and eigenvectors - note that evs contains omega^2!
				evsSq, evecs, _ = FDM_FindEvals(N, theta, alpha3, lOff, nEvals=nEvals, checks=False, saveEvals=False, saveEvecs=False)
								
				# write the output, including whether or not we converged
				writeOut = np.hstack((np.array([t1, tFixed], dtype=float), np.sqrt(evsSq)))
				# write the data to the output
				with open(fName, 'ab') as f:
					np.savetxt(f, [writeOut], delimiter=',')
					
				# write the eigenfunctions out if we also wanted them
				if saveFuncs:
					for n in range(np.shape(evecs)[1]):
						funcWriteOut = np.hstack((np.array([t1, tFixed]), evecs[:,n]))
						with open(funcFilename, 'ab') as f:
							np.savetxt(f, [funcWriteOut], delimiter=',')
		else:
			# dimSpecs[1] must be 1 in that case, so loop over theta2, theta1 = tFixed
			print('1D loop over theta_2, theta_1 fixed at %.3f' % (tFixed*pi))
			for j, t2 in enumerate(unifRange):
				# Inform about progress if requested
				if (not lOff):
					print( ' << Beginning: (%d) of (%d) >>' % (j,nPts-1))
				# Generate QM vector
				theta = np.array([tFixed, t2], dtype=float) * pi
				# compute eigenvalues and eigenvectors - note that evs contains omega^2!
				evsSq, evecs, _ = FDM_FindEvals(N, theta, alpha3, lOff, nEvals=nEvals, checks=False, saveEvals=False, saveEvecs=False)
								
				# write the output, including whether or not we converged
				writeOut = np.hstack((np.array([t1, tFixed], dtype=float), np.sqrt(evsSq)))
				# write the data to the output
				with open(fName, 'ab') as f:
					np.savetxt(f, [writeOut], delimiter=',')
					
				# write the eigenfunctions out if we also wanted them
				if saveFuncs:
					for n in range(np.shape(evecs)[1]):
						funcWriteOut = np.hstack((np.array([t1, tFixed]), evecs[:,n]))
						with open(funcFilename, 'ab') as f:
							np.savetxt(f, [funcWriteOut], delimiter=',')
	else:
		# do a full, 2D loop
		print('2D loop consisting of %d points (%d solves required)' % (nPts*nPts, nPts*nPts*N))
		for i, t1 in enumerate(unifRange):
			for j, t2 in enumerate(unifRange):
				# Inform about progress if requested
				if (not lOff):
					print( ' << Beginning: (%d,%d) of (%d,%d) >>' % (i,j,nPts-1,nPts-1))
				# Generate QM vector
				theta = np.array([t1, t2], dtype=float) * pi
				# compute eigenvalues and eigenvectors - note that evs contains omega^2!
				evsSq, evecs, _ = FDM_FindEvals(N, theta, alpha3, lOff, nEvals=nEvals, checks=False, saveEvals=False, saveEvecs=False)
								
				# write the output, including whether or not we converged
				writeOut = np.hstack((np.array([t1, tFixed], dtype=float), np.sqrt(evsSq)))
				# write the data to the output
				with open(fName, 'ab') as f:
					np.savetxt(f, [writeOut], delimiter=',')
					
				# write the eigenfunctions out if we also wanted them
				if saveFuncs:
					for n in range(np.shape(evecs)[1]):
						funcWriteOut = np.hstack((np.array([t1, tFixed]), evecs[:,n]))
						with open(funcFilename, 'ab') as f:
							np.savetxt(f, [funcWriteOut], delimiter=',')
							
	# we don't actually have anything to return here, since SciPy's eigs will error if we don't converge anyway
	return

#%% Command-line execution

if __name__=='__main__':

	parser = argparse.ArgumentParser(description='Computation of eigenvalues and eigenvectors of the Composite measure problem in the Cross-in-plane geometry, via Finite Difference approximation.')
	parser.add_argument('-fn', default='', type=str, help='Output file name, if blank filename will be auto-generated')
	parser.add_argument('-fd', default='./FDM_Results/', type=str, help='Directory to save output file to.')
	parser.add_argument('-nPts', default=11, type=int, help='[Default 11] Number of gridpoints in each dimension of the quasi-momentum to use.')
	parser.add_argument('-N', default=31, type=int, help='[Default 31] Number of spatial gridpoints in each dimension to use in the Finite difference mesh. Must be odd.')
	parser.add_argument('-nEvals', default=2, type=int, help='[Default 2] Number of eigenfunctions and eigenvalues to find (starting from lowest eigenvalue)')
	parser.add_argument('-a', default=0., type=float, help='[Default 0.] Value of coupling constant at central vertex.')
	parser.add_argument('-lOff', action='store_true', help='Suppress printing of progress and solver log to the screen')
	parser.add_argument('-funcs', action='store_true', help='Write out the eigenfunctions that are found, as well as the eigenvalues')
	parser.add_argument('-oneD', action='store_true', help='Only loop over theta values in one dimension. Default will be dimension 0, using theta2=-pi')
	parser.add_argument('-tDim', default=0, type=int, help='[Default 0] If looping over one dimension in theta, specifies the dimension to loop over')
	parser.add_argument('-tFixed', default=0, type=int, help='[Default 0] If looping over one dimension in theta, sets the value of the fixed QM value to be np.linspace(-1,1,nPts)[tFixed]*pi')

	# extract input arguments and get the setup ready
	args = parser.parse_args()	
	
	# first, require N to be odd!
	if args.N % 2 == 0:
		raise ValueError('Expected N odd, but got %d' % args.N)
	 
	# Output file handles
	if (not args.fn):
		# if no filename was given, auto-generate
		resultsFile = args.fd + 'FDMEvals-N-%d.csv' % (args.N)
	else:
		resultsFile = args.fd + args.fn
	
	# catch 1D loops with bad indices for the fixed component of the QM
	if (args.oneD and (args.tFixed>=args.nPts) or (args.tFixed<0)):
		raise ValueError('Invalid index fix at tFixed')
	dSpec = [args.oneD, args.tDim, args.tFixed]
	
	# Perform loop at these variable values
	FDM_ThetaLoop(args.nPts, resultsFile, N=args.N, nEvals=args.nEvals, alpha3=args.a, lOff=args.lOff, saveFuncs=args.funcs, dimSpecs=dSpec)
	
	# if we got to here then we succeeded at everything we tried
	sys.exit(0)