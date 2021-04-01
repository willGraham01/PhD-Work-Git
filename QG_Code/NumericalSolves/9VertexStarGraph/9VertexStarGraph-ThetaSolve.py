#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 14:36:35 2021

@author: will

Script and related functions for solving the 9-vertex star-shaped QG problem suggested by Kirill.

Given Lambda = \sqrt(omega^2-kappa^2), the DR for a given (theta0, theta1) and vertex constant alpha is

0 = -2 * Lambda * sin(Lambda) * cos^2(t0/2) * cos^2(t1/2)
	+ Lambda * (cos(Lambda) * sin(Lambda*sqrt(2)) + sin(Lambda)*cos(Lambda*sqrt(2)))
	+ Lambda * (sin(Lambda) + sin(Lambda*sqrt(2)))
	- Lambda * sin(Lambda*sqrt(2)) * (cos^2(t0/2) + cos^2(t1/2))
	- (alpha*omega^2/4) * sin(Lambda) * sin(Lambda*sqrt(2)) 	(*)
		
For a given alpha, kappa, and omega, we can attempt to solve for a value of theta that satisfies (*), then repeat this process across a suitable grid in (omega, kappa).
This is likely faster than the exhaustive binary-search method along the axes of our grid, and should provide exact answers as to when, and when not, a given (omega, kappa) is part of the spectrum.

Throughout, we will work in v0 = cos^2(theta0/2) and v1 = cos^2(theta1/2), which makes the equation we have to solve a simple polynomial in the variable v = [v0, v1], and we seek solutions where v \in [0,1]^2.
"""

import sys
sys.path.append('../../')

import numpy as np
from numpy import sin, cos, sqrt

import AuxPlotFns as APF #FastFunctionPlot(x,f) and FigWindowSettings(ax) are very useful shortcuts

import time

def CreateHandles(omega, kappa, alpha):
	'''
	Create function handle for the RHS of (*) as a function of v.
	Also create function handle for the gradient (wrt v) of the RHS of (*) as a function of v.
	'''
	
	L = sqrt(omega*omega - kappa*kappa)
	Lsqrt2 = L * sqrt(2.)
	
	#f(v) = A*v[0]*v[1] + B(v[0] + v[1]) + C, where
	A = -2. * L * sin(L)
	B = -1. * L * sin(Lsqrt2)
	C = L * ( cos(L)*sin(Lsqrt2) + sin(L)*cos(Lsqrt2) + sin(L) + sin(Lsqrt2) ) - (alpha*omega*omega/4.) * sin(L) * sin(Lsqrt2)
	
	def f(v): 
		'''
		Generated function from CreateHandles for RHS of (*), input should be of shape (2,) and consist of the point v=[v0,v1] to evaluate the function at.
		Function is vectorised, so an input of shape (2,n) will interpret each column as one set of points v, as above.
		'''
		return (A * np.prod(v, axis=0)) + (B * np.sum(v, axis=0)) + C
	#df(v) = [A*v[1]+B, A*v[0] + B], so need to swap the rows of v, multiply by A, then add B
	def df(v):
		'''
		Generated function from CreateHandles for gradient of (*), input should be of shape (2,) and consist of the point v=[v0,v1] to evaluate the function at.
		Function is vectorised, so an input of shape (2,n) will interpret each column as one set of points v, as above.
		'''
		return (A * v[[1,0]]) + B
	#in case we want to have the coefficients to hand for debugging
	constants = np.asarray([A,B,C])
	
	return f, df, constants

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

def ValidateSolutions(regionGrid, solStore, kappaRange, omegaRange, alpha, tol=1e-8):
	'''
	Having run through this theta-mesh and found what we deem to be "solutions", it'd be a piece of mind to cross-check that they are, in fact, actually solving (*)!
	'''
	
	solIndices = np.argwhere(regionGrid) #solIndices is an (n,2) array storing indices of all the non-zero (so True) elements of regionGrid. Each row [k,w] corresponds to kappaRange[k], omegaRange[w]
	kIndices = solIndices[:,0]
	wIndices = solIndices[:,1]
	nSols = np.shape(kIndices)[0]
	vSols = solStore[ regionGrid == True ].T #vSols is (2,n) array storing the solutions v = [v0, v1] along columns

	badCount = 0
	for i in range(nSols):
		#for each ``solution", check if f(v) = 0, or close enough
		w = omegaRange[wIndices[i]]
		k = kappaRange[kIndices[i]]
		f = CreateHandles(w, k, alpha)[0]
		if np.abs(f(vSols[:,i])) >= tol:
			print('This does not appear to be a solution!')
			print('omega, kappa, f(v): %.5f, %.5f, %.5e' % (w, k, f(vSols[:,i])))
			print('vSols[:,i]:', vSols[:,i])
			badCount += 1
	print('Found %d bad solutions' % (badCount))	
	return

if __name__ == "__main__":
	
	alpha = -2.
	wPiBands = 3
	wPtsPerPi = 1000
	omegaRange = np.linspace(0, wPiBands*np.pi, num=wPtsPerPi*wPiBands)
	kPtsPerPi = 500
	kPiBands = 3
	kappaRange = np.linspace(0, kPiBands*np.pi, num=kPtsPerPi*kPiBands, endpoint=False) #endpoint=False ensures that there is at least ONE kappa value to try per omega value, if the endpoints of both ranges are the same

	dstr = APF.GetDateTime() #for saving with a timestamp
	print('--- Setting up plot stores')
	regionGrid = np.zeros((np.shape(kappaRange)[0], np.shape(omegaRange)[0]), dtype=bool) #to create plot
	#debug and run info variables, store solutions v in grid form too
	solStore = np.zeros((np.shape(kappaRange)[0], np.shape(omegaRange)[0], 2), dtype=float) 
	noCon = 0
	yesCon = 0
	boundCon = 0
	
	print('--- Solving across omega, kappa grid')
	startTime = time.time()
	for k, kappa in enumerate(kappaRange):
		print('%d, ' % (k), end='')
		if (k+1)%15==0:
			print('\n', end='')
		#we don't need to look at those values in omegaRange for which omegaRange < kappa, since these are forbidden.
		validOmega = omegaRange[ omegaRange >= kappa ]
		offSet = np.argmax( omegaRange >= kappa) #omegaRange[offSet] = validOmega[0] 
		for w, omega in enumerate(validOmega):
			v0 = np.ones((2,), dtype=float)*0.5 #start at [1/2, 1/2], since we want to try and converge to the closest solution in [0,1]^2. This could be made less ignorant with some thought
			f, df = CreateHandles(omega, kappa, alpha)[0:2] #remember slice index 2 is not included!
			vSol, solCheck = NewtonSolve(v0, f, df) #solve for v via Newton
			#validate solution
			if solCheck:
				#if we converged, we still might have only found a solution for v outside of [0,1]^2, so we need to check this too
				if (( vSol[0] >= 0. ) and (vSol[0] <= 1.)):
					#component vSol[0] is in the right region
					if (( vSol[1] >= 0. ) and (vSol[1] <= 1.)):
						#component vSol[1] is in the right region too
						yesCon += 1
						regionGrid[k, w + offSet] = True
						solStore[k, w + offSet] = np.copy(vSol)
					else:
						#vSol[1] is outside the required region
#						print('vSol[1] is outside correct region: omega, kappa, vSol')
#						print('%.5f, %.5f' % (omega, kappa), end=', ')
#						print(vSol)
						boundCon += 1
				elif (( vSol[1] >= 0. ) and (vSol[1] <= 1.)):
					#component vSol[1] is in the right region, but vSol[0] isn't
#					print('vSol[0] is outside correct region: omega, kappa, vSol')
#					print('%.5f, %.5f' % (omega, kappa), end=', ')
#					print(vSol)
					boundCon += 1
				else:
					#both components are outside the required region
#					print('vSol[0] and vSol[1] are outside correct region: omega, kappa, vSol')
#					print('%.5f, %.5f' % (omega, kappa), end=', ')
#					print(vSol)
					boundCon += 1					
			else:
				#no convergence, so something was up, let's assume there wasn't a zero in this region
#				print('Newton solve did not converge at (omega, kappa): (%.5f, %.5f)' % (omega, kappa))
				noCon += 0
	endTime = time.time()
	print("Grid time: %s seconds" % (time.time() - startTime))
	print('Finished Solve, found #of: \n Solutions: %d \n No-convergence: %d \n Out-of-bounds convergence: %d' % (yesCon, noCon, boundCon))
				
	#we now plot regionGrid as a surface...
	print('--- Creating plot...')
	plotTime = time.time()
	alphaStr = r'alpha=' + r"{:.2f}".format(alpha)
	titStr = r'Dispersion relation, ' + APF.GenTitStr(0, alpha) #setting wavenumber=0 will only make a title for alpha
	fig, ax = APF.PlotContour(omegaRange/np.pi, kappaRange/np.pi, np.array(regionGrid, dtype=int))
	ax.set_xlabel(r'$\frac{\omega}{\pi}$')
	ax.set_ylabel(r'$\frac{\kappa}{\pi}$')
	ax.set_title(titStr)
#	fig.savefig(dstr + r'-DR_' + alphaStr + r'.png', bbox_inches='tight')
	fig.savefig(dstr + r'-DR_' + alphaStr + r'.pdf', bbox_inches='tight')
	fig.show()
	plotEndTime = time.time()
	print('--- Plot completed')
	
	print('--- Validating...')
	ValidateSolutions(regionGrid, solStore, kappaRange, omegaRange, alpha, tol=1e-8)
	print('--- Run end')
	
				