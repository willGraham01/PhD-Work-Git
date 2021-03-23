#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 14:36:35 2021

@author: will

Script and related functions for solving the 9-vertex star-shaped QG problem suggested by Kirill.

Given Lambda = \sqrt(omega^2-kappa^2), the DR for a given (theta0, theta1) and vertex constant alpha is

0 = cos(Lambda/2)sin(Lambda/2)cos^2(t0/2)cos^2(t1/2)
	- sin(Lambda/sqrt(2))cos(Lambda/sqrt(2))/2 * (cos^2(t0/2)+cos^2(t1/2)
	+ cos(Lambda/sqrt(2))cos(Lambda/2)sin(Lambda*(1+sqrt(2))/2)
	- alpha*omega^2/(4*Lambda) * sin(Lambda/2)cos(Lambda/2)sin(Lambda/sqrt(2))cos(Lambda/sqrt(2))       (*)
		
Of course, we should multiply out the factor of 1/Lambda to ensure stability in this script.

For a given alpha, kappa, and theta, we can solve and obtain a value for omega.
We then repeat the process, varying theta over a mesh, and storing all the values of omega that we find.
"""

import sys
sys.path.append('../../')

import numpy as np
from numpy import sin, cos, sqrt

import matplotlib.pyplot as plt

import AuxPlotFns as APF #FastFunctionPlot(x,f) and FigWindowSettings(ax) are very useful shortcuts

from scipy.optimize import fsolve, bisect

import time

def RHS(omega, alpha, kappa, theta):
	'''
	Evaluates the function that forms Lambda * the RHS of the equation (*) in the description above.
	'''
	
	L = sqrt(omega*omega - kappa*kappa) #compute lambda, ensure vectorisation
	L2 = L / 2.
	Lsqrt2 = L / sqrt(2.) #because we use these more often than we use L
	
	val = cos(L2) * sin(L2) * ( ( cos(theta[0]/2.) * cos(theta[1]/2.) )**2 )
	val += - ( sin(Lsqrt2) * cos(Lsqrt2) / 2. ) * ( cos(theta[0]/2.)*cos(theta[0]/2.) + cos(theta[1]/2.)*cos(theta[1]/2.) )
	val += cos(Lsqrt2) * cos(L2) * sin(L*(1.+sqrt(2.))/2.)
	val += - alpha*(omega*omega)/(4*L) * sin(L2) * cos(L2) *sin(Lsqrt2) * cos(Lsqrt2)
	
#	val = L * ( sin((1.+sqrt(2.))*L) / 4. )
#	val += - ( (alpha*omega*omega) / 16. ) * sin(L) * sin(sqrt(2.)*L)
#	val += - ( np.sum(cos(theta)) / 8. ) * L * sin(sqrt(2.)*L)
#	val += ( ( np.product(cos(theta)) + np.sum(cos(theta)) - 3. ) / 8. ) * L * sin(L)
	
	return val

def FindSignChanges(array):
	'''
	Given an array, return an an array of ones and zeros determining whether a sign change has occurred between two subsequent values in the array.
	INPUTS:
		array: (n,) float numpy array, for which sign changes between subsequent entries are to be determined
	OUTPUTS:
		signChange: (n,) bool numpy array, signChange[i] = True implies that signChange[i] and signChange[i-1] have different signs
	'''
	
	asign = np.sign(array)
	signChange = ((np.roll(asign, 1) - asign) != 0).astype(bool)
	sz = asign == 0
	while sz.any():
		asign[sz] = np.roll(asign, 1)[sz]
		sz = asign == 0
	signChange[0] = 0 #no comparison from last element to 1st
	
	return signChange

def SolveAtStartingVals(startingVals, alpha, kappa, theta, filter=True, biSearch=True):
	'''
	Solves RHS(omega) = 0 for the given input values of alpha, kappa, and theta.
	There will be one solve for each element of startingVals, and these will be used as the subsequent starting guesses for each solve.
	
	If the optional argument filter is true, negative omega solutions will be removed before returning, and np.unquie will be run on the array just in case there are any duplicate roots to within machine precision.
	
	If the optional argument biSearch is False, then we will run one solve per value in startingVals, using each value as the starting guess for one solve. If True, we will first use the points provided in startingVals to determine regions in which RHS changes sign, then run an interval bisection method on those intervals.
	'''
	
	#create lambda function to call RHS
	f = lambda omega: RHS(omega, alpha, kappa, theta)
	
	if biSearch:
		#interval bisection approach, first find all the sign changes of f
		useVals = startingVals[ startingVals >= kappa ] #remember that omega < kappa is forbidden!
		
		fVals = f(useVals)
		signChanges = FindSignChanges(fVals)
		changeIndices = np.nonzero(signChanges)[0]
		
		#now use scipy.bisect to find roots in the intervals which have sign changes in them
		expectedRoots = np.shape(changeIndices)[0]
		solns = np.zeros((expectedRoots), dtype=float)
		for i, valIndex in enumerate(changeIndices):
			#there was a change of sign in fVals between startingVals[i-1] and startingVals[i], so we should locate a root
			try:
				solns[i] = bisect(f, useVals[valIndex-1], useVals[valIndex])
				if np.isnan(solns[i]): 
					raise ValueError('Solution is NaN')
				if not np.isclose(f(solns[i]), 0.0):
					raise ValueError('Solution not close to 0.0')
			except ValueError:
				print('Warning: bisect encountered an error, searching in interval [%.5f, %.5f]' % (useVals[valIndex-1], useVals[valIndex]))
				print('Solution omega / f(omega): %.5f / %.5f' % (solns[i], f(solns[i])))
				solns[i] = 0.0 #temp just overwrite the bad values to 0, then they'll get cut off if filtered out
	else:
		#store for solution values
		solns = np.zeros_like(startingVals[ startingVals >= kappa ])
		
		#solve for each starting value and record the output
		for i,w0 in enumerate(startingVals[ startingVals >= kappa ]):
			#we need to catch errors if they occur
			try:
				solns[i] = fsolve(f, w0)
				if np.isnan(solns[i]): 
					raise ValueError('Solution is NaN')
				if not np.isclose(f(solns[i]), 0.0):
					raise ValueError('Solution not close to 0.0')
			except ValueError:
				print('Warning: fsolve encountered an error, starting guess was %.5f' % w0)
				print('Solution omega / f(omega): %.5f / %.5f' % (solns[i], f(solns[i])))
				solns[i] = 0.0 #temp just overwrite the bad values to 0, then they'll get cut off if filtered out
	
	if filter:
		solns = solns[solns >= 0.]
		solns = np.unique(solns)
	
	return solns

def GridSweep(savefile, t0Vals, t1Vals, startingVals, alpha, kappa, **kwargs):
	'''
	For the given starting values, compute roots for each quasimomentum in t0Vals, t1Vals, then return the results.
	
	**kwargs:
		filter: bool, passed to filter argument of SolveAtStartingVals
		biSearch: bool, passed to biSearch argument of SolveAtStartingVals
	'''

	sF = open(savefile, 'a')
	solCount = 0
	for t0 in t0Vals:
		for t1 in t1Vals:
			solns = SolveAtStartingVals(startingVals, alpha, kappa, np.asarray([t0,t1]), **kwargs)
			np.savetxt(sF, solns, delimiter=',', newline='\n')
			solCount += np.shape(solns)[0]
	sF.close()
	
	return solCount

def CompareTimings(soloRun = False):
	'''
	Manual timing test between the binary search method and the straight-up fsolve method
	As a bi-product, this will also produce two solution files from each method, which can also be compared if desired
	'''
	
	nPiBands = 3
	ptsPerPi = 100
	startingVals = np.linspace(0,nPiBands*np.pi,nPiBands*ptsPerPi)
	alpha = 3. #alpha close to zero seems to cause bad (numerical) behaviour
	kappa = 0.
	t0Vals = np.linspace(-np.pi, np.pi, num=5, endpoint=False)
	t1Vals = np.linspace(-np.pi, np.pi, num=5, endpoint=False)
	
	if soloRun:
		theta = np.array([np.random.choice(t0Vals), np.random.choice(t1Vals)], dtype=float)
		
		print('Solo run: theta [%.5f, %.5f], kappa %.3f, alpha %.3f' % (theta[0], theta[1], kappa, alpha))
		
		startTime = time.time()
		solnsFsolve = SolveAtStartingVals(startingVals, alpha, kappa, theta, filter=True, biSearch=False)
		print("Solo run: fsolve: --- %s seconds ---" % (time.time() - startTime))
		
		startTime = time.time()
		solnsBisect = SolveAtStartingVals(startingVals, alpha, kappa, theta, filter=True, biSearch=True)
		print("Solo run: bisect: --- %s seconds ---" % (time.time() - startTime))
	
		f = lambda omega: RHS(omega, alpha, kappa, np.zeros((2)))
		fig, ax = plt.subplots(1)
		APF.FigWindowSettings(ax)
		ax.plot(startingVals, f(startingVals))
		ax.scatter(solnsFsolve, np.zeros_like(solnsFsolve), s=4, c='r', marker='x')
		ax.scatter(solnsBisect, np.zeros_like(solnsBisect), s=4, c='g', marker='o')
		fig.show()
	else:
		dstr = APF.GetDateTime()
		fileNameF = dstr + 'solnOuts-fsolve.csv'
		fileNameB = dstr + 'solnOuts-bisect.csv'
		
		print('Grid sweep; temp files created:')
		print('fsolve: ', fileNameF)
		print('bisect: ', fileNameB)
		
		
		startTime = time.time()
		solCountF = GridSweep(fileNameF, t0Vals, t1Vals, startingVals, alpha, kappa, biSearch=False)
		print("Grid sweep: fsolve: --- %s seconds ---, %d solutions" % (time.time() - startTime, solCountF))
		
		startTime = time.time()
		solCountB = GridSweep(fileNameB, t0Vals, t1Vals, startingVals, alpha, kappa, biSearch=True)
		print("Grid sweep: bisect: --- %s seconds ---, %d solutions" % (time.time() - startTime, solCountB))
		
		foundSolsF = np.loadtxt(fileNameF, delimiter=',')
		foundSolsB = np.loadtxt(fileNameB, delimiter=',')
		
		fig, ax = plt.subplots(1)
		APF.FigWindowSettings(ax)
		ax.scatter(foundSolsF/np.pi, np.ones_like(foundSolsF), s=1, c='r', marker='x')
		ax.scatter(foundSolsB/np.pi, -1. * np.ones_like(foundSolsB), s=1, c='g', marker='o')
		ax.set_xlim(np.min(startingVals/np.pi), np.max(startingVals/np.pi))
		ax.set_xlabel(r'$\frac{\omega}{\pi}$')
		fig.show()
		
	return

def RootSearchPlot(omegaVals, kappaVals, t0Vals, t1Vals, alpha=1.):
	
	dstr = APF.GetDateTime() #for saving with a timestamp	
	regionGrid = np.zeros((np.shape(kappaVals)[0], np.shape(omegaVals)[0]), dtype=bool)
	
	print('----Beginning root search')
	print('kappa index: ', end='')
	startTime = time.time()
	for k, kappa in enumerate(kappaVals):
		print('%d,' % (k), end=' ')
		if (k+1)%10 == 0:
			print('\n', end='')
		if kappa >= np.max(omegaVals):
			print('kappa value has exceeded max(omegaVals) - this portion of the plot will be entirely zero')
			break #this should avoid the issue with kappaVals final entry being equal to omegaVals final entry, and thus causing empty arrays to appear and thus infinite loops below
		useVals = omegaVals[ omegaVals >= kappa ] #remember that omega < kappa is forbidden!
		indOffset = np.argmax( omegaVals >= kappa ) #this provides us with the first index in omegaVals that corresponds to index 0 of useVals
		for t0 in t0Vals:
			for t1 in t1Vals:
				theta = np.array([t0, t1], dtype=float)
				f = lambda omega: RHS(omega, alpha, kappa, theta)
				
				#find regions in which roots lie
				fVals = f(useVals)
				signChanges = FindSignChanges(fVals)
				changeIndices = np.nonzero(signChanges)[0]
				#now account for the offset that we had before
				changeIndices += indOffset
				
				#now set every value in the regionGrid which corresponds to changeIndices, kappaVal[k] to be True
				regionGrid[k, changeIndices] = True
	#after completing this loop (if it ever completes...), regionGrid now details the regions in which RHS changes sign, and so there is at least one root in this region. We now just take the midpoints (in omega) as the roots, and plot a 0-1 surface to approximate the dispersion relation.
	endTime = time.time()
	print('----Finished root-search')
	
	print('----Plotting')
	surfGrid = regionGrid[:, 1:]
	#regionGrid indicates if there was a root between omegaVals[i] and omegaVals[i-1].
	omegaMidPts = (omegaVals[1:] + omegaVals[:-1]) / 2.
	#we treat roots as being at the value omegaVals[i]+omegaVals[i-1]/2
	#Therefore, if regionGrid[k, i] is True (i>0), then there is a root at omegaMidPoints[i] at the value kappaVals[k]. 
	#Namely, if surfGrid[i] is True, then there is a root at kappaVals[k], omegaMidPts[i]
	
	#we now plot surfGrid as a surface...
	print('Creating plot...')
	plotTime = time.time()
	titStr = r'Dispersion relation, ' + APF.GenTitStr(0, alpha) #setting wavenumber=0 will only make a title for alpha
	fig, ax = APF.PlotContour(omegaMidPts/np.pi, kappaVals/np.pi, np.array(surfGrid, dtype=int))
	ax.set_xlabel(r'$\frac{\omega}{\pi}$')
	ax.set_ylabel(r'$\frac{\kappa}{\pi}$')
	ax.set_title(titStr)
	fig.savefig(dstr + '-DR_alpha=1.png', bbox_inches='tight')
	fig.savefig(dstr + '-DR_alpha=1.pdf', bbox_inches='tight')
	fig.show()
	plotEndTime = time.time()
	print('----Plot completed')
	
	print('----Run info dump:')
	print('Root search time: %s seconds' % (endTime - startTime))
	print('Time / kappa: %s seconds' % ((endTime - startTime) / kPts) )
	print('Plot time: %s seconds' % (plotEndTime - plotTime) )
	print('----End Run')
	
	return fig, ax

def NewtonSolve(x0, f, df, nIt=100, tol=1e-8):
	'''	
	Find root of a vector-to-scalar function using Newton's method. This is a workaround for Scipy's lack of vector-to-scalar solve functions. See here for details and useage warnings:
	https://stackoverflow.com/questions/24189424/best-way-to-find-roots-of-a-multidimensional-scalar-function-with-scipy
	In particular, the only useage intention for this function is in ThetaSolve, where we are solving a 2D-polynomial for a root!
	
	INPUTS:
		x0: (n,) float numpy array, initial guess for solution
		f: 	lambda function, must take a vector of the same shape as x0 as an input, and evaluates the function whose root we seek
		df: 	lambda function, must take a vector of the same shape as x0 as an input, and evaluates the Jacobian of the function whose root we seek
		nIt: 	(optional) int, max number of iterations to perform. Default 100
		tol: 	(optional) float, tolerance of root. Default 1e-8
	OUTPUTS:
		x: 	(n,) float numpy array, approximation to solution
		converged: 	bool, if True, a root was found to the tolerance of tol, otherwise False
	'''
	
	x = np.copy(x0)
	converged = False
	for n in range(nIt): #perform max nIt iterations
		fVal = f(x)
		dfVal = df(x)
		
		if np.abs(fVal) < tol: #break if close to root
			converged = True
			break
		else: #Newton step
			x = x - dfVal * fVal / (np.linalg.norm(dfVal)**2)
	return x, converged

def ThetaSolve(omega, kappa, alpha):
	'''
	The reverse-thinking approach to finding spectral points - given an omega, kappa, we "solve" for theta to see if (*) has a root - if it does, then (omega, kappa) is an eigenvalue.
	Otherwise, there shouldn't be any convergence and so we deem this a failure.
	'''
	
	L = sqrt(omega*omega - kappa*kappa) #compute lambda, ensure vectorisation
	L2 = L / 2.
	Lsqrt2 = L / sqrt(2.) #because we use these more often than we use L	

	A = cos(L2) * sin(L2)
	B = - sin(Lsqrt2) * cos(Lsqrt2)
	C = cos(Lsqrt2) * cos(L2) * sin(L * (1+sqrt(2.))/2.) 
	C += - (alpha*omega*omega/(4.*L)) * sin(Lsqrt2) * cos(Lsqrt2) * sin(L2) * cos(L2)
	
	#we now determine whether there is at least one solution to
	# 0 = Axy + B(x+y) + C, 
	#where 0 <= x,y <= 1 (x,y are cos^2(theta0) and cos^2(theta1), respectively)
	
	f = lambda v: A*np.prod(v) + B*np.sum(v) + C
	df = lambda v: np.array( [A*v[1] + B, A*v[0] + B] )
	x0 = np.array([0.5, 0.5]) #ignorant starting guess, but we want to converge between 0 and 1, so this is the best guess if we think we'll find the "closest" root
	
	v, converged = NewtonSolve(x0, f, df)
	solution = False
	if not converged:
		print('Newton Solve did not converge, no root found!')
	else:
		if (v[0] >= 0. and v[0] <= 1.):
			if (v[1] >= 0. and v[1] <= 1.):
				print('Theta values found, (omega, kappa) is an eigenpair: (%.3f, %.3f)' % (omega, kappa))
				solution = True
			else:
				print('Theta value found, but theta1 is not between 0 and 1')
		else:
			print('Theta value found, but theta0 is not between 0 and 1')
	return v, solution

if __name__ == "__main__":
	
	alpha = 1.
	nPiBands = 2
	ptsPerPi = 1000
	tPts = 100
	kPts = 100	#estimate that, with ptsPerPi = 1000, tPts = 100, that the time for one sweep of a fixed kappa value is 2.6 seconds.
	dstr = APF.GetDateTime() #for saving with a timestamp
	
	omegaVals = np.linspace(0, nPiBands*np.pi, nPiBands*ptsPerPi)
	kappaVals = np.linspace(0, nPiBands*np.pi, kPts, endpoint=False)
	t0Vals = np.linspace(-np.pi, np.pi, num=tPts, endpoint=False)
	t1Vals = np.linspace(-np.pi, np.pi, num=tPts, endpoint=False)

	#fig, ax = RootSearchPlot(omegaVals, kappaVals, t0Vals, t1Vals, alpha=alpha)
	
	regionGrid = np.zeros((np.shape(kappaVals)[0], np.shape(omegaVals)[0]), dtype=bool)
	
	print('----Beginning grid theta-solves')
	print('omegaVals index: ', end='')
	startTime = time.time()
	for w, omega in enumerate(omegaVals):
		print('%d,' % (w), end=' ')
		if (w+1)%10 == 0:
			print('\n',end='')
			for k, kappa in enumerate(kappaVals[ kappaVals >= omega ]):
				#this will cut-off the values of kappa that are less than the value of omega here...
				indOffset = np.argmax( kappaVals >= omega ) #this provides us with the first index in kappaVals that corresponds to index 0 of kappaVals[ kappaVals >= omega ]
				
				x, solution = ThetaSolve(omega, kappa, alpha)
				if solution:	
					regionGrid[k + indOffset, w] = True #found a solution, set the grid to be equal to zero
	endTime = time.time()
	print('----Finished grid theta-solves')
	
	print('----Plotting')
	surfGrid = regionGrid[:, 1:]
	#regionGrid indicates if there was a root between omegaVals[i] and omegaVals[i-1].
	omegaMidPts = (omegaVals[1:] + omegaVals[:-1]) / 2.
	#we treat roots as being at the value omegaVals[i]+omegaVals[i-1]/2
	#Therefore, if regionGrid[k, i] is True (i>0), then there is a root at omegaMidPoints[i] at the value kappaVals[k]. 
	#Namely, if surfGrid[i] is True, then there is a root at kappaVals[k], omegaMidPts[i]
	
	#we now plot surfGrid as a surface...
	print('Creating plot...')
	plotTime = time.time()
	titStr = r'Dispersion relation, ' + APF.GenTitStr(0, alpha) #setting wavenumber=0 will only make a title for alpha
	fig, ax = APF.PlotContour(omegaMidPts/np.pi, kappaVals/np.pi, np.array(surfGrid, dtype=int))
	ax.set_xlabel(r'$\frac{\omega}{\pi}$')
	ax.set_ylabel(r'$\frac{\kappa}{\pi}$')
	ax.set_title(titStr)
	fig.savefig(dstr + '-TS-DR_alpha=1.png', bbox_inches='tight')
	fig.savefig(dstr + '-TS-DR_alpha=1.pdf', bbox_inches='tight')
	fig.show()
	plotEndTime = time.time()
	print('----Plot completed')
	
	print('----Run info dump:')
	print('Root search time: %s seconds' % (endTime - startTime))
	print('Time / omega: %s seconds' % ((endTime - startTime) / np.shape(omegaVals)[0]) )
	print('Plot time: %s seconds' % (plotEndTime - plotTime) )
	print('----End Run')