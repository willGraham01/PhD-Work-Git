#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 18:29:32 2020

@author: will

This file contains various functions that I may wish to employ across multiple files, scripts and potentially integrate into classes for my PhD.

Rough table of contents:
	- csc
	- cot
	- AbsLess
	- RemoveDuplicates
	- UnitVector
	- CompToReal
	- RealToComp
	- PolyEval
	- NLII
"""

from warnings import warn

import numpy as np
from numpy import sin, tan
from numpy.linalg import norm

from scipy.optimize import fsolve

def csc(x):
	'''
	Cosecant function, derived from np.sin
	INPUTS:
		x: 	(array-like), values to evaluate csc at
	OUTPUTS:
		cscx: 	numpy array, cosecant of each value in x
	'''
	return 1/ sin(x)

def cot(x):
	'''
	Cotangent function, derived from np.tan
	INPUTS:
		x: 	(array-like), values to evaluate cot at
	OUTPUTS:
		cotx: 	numpy array, cotangent of each value in x
	'''
	return 1/ tan(x)

def AbsLess(x, a, b=np.NaN, strict=False):
	'''
	Checks whether (each value in an array) x lies between two values. If only one value is supplied, it checks whether abs(x) is less than this value.
	INPUTS:
		x: 	(n,) float numpy array, values to be tested
		a: 	float, if b is not supplied then the check abs(x)<=a is performed
		b: 	(optional) float, if supplied then a<b is required, and the check a<=abs(x)<=b is performed
		strict: 	(optional) bool - default False, if True then all non-strict inequalities are changed to strict inequalities in the above documentation.
	OUTPUTS:
		tf: 	(n,) bool numpy array, logical index array corresponding to points that satisfied the requested check
	'''
	
	if np.isnan(b):
		#optional argument b not supplied, just check abs(x)<= or < a depending on strict
		if strict:
			#check abs(x)<a
			tf = abs(x) < a
		else:
			#check abs(x) <= a
			tf = abs(x) <= a
	elif a<b:
		#optional argument b supplied and valid, checking a - abs(x) - b depending on strict
		offSet = 0.5*(b+a)
		bound = 0.5*(b-a)
		if strict:
			tf = abs(x - offSet) < bound
		else:
			tf = abs(x - offSet) <= bound
	else:
		raise ValueError('Argument b supplied but is not strictly greater than than a.')
	
	return tf

def RemoveDuplicates(valIn, tol=1e-8):
	'''
	Remove duplicate entries from a given list or numpy array of complex values; classing duplicates to be be those values whose norm difference is less than a given tolerance.
	INPUTS:
		valIn: 	list or 1D numpy array, complex values to be sorted for duplicates
		tol: 	(optional) float - default 1e-8, tolerance within which values are classed as duplicates
	OUTPUTS:
		unique: 	complex numpy array, the unique values using the mean of all values determined to be "the same"
		uIDs: 	int numpy array, if vals[i] is deemed a duplicate of vals[j], then uIDs[i]=uIDs[j], essentially returning the "groupings" of the original values.
	'''
	
	vals = np.asarray(valIn) #in case the input is a list, this will allow us to do logic slicing with numpy arrays
	nVals = np.shape(vals)[0]
	uIDs = np.zeros((nVals,), dtype=int)
	newID = 1
	
	for currInd in range(nVals):
		#has this value already be assigned a group?
		if uIDs[currInd]==0:
			#this value does not have a group yet - give it one, then go through and find it's other members
			uIDs[currInd] = newID
			newID += 1 #update the group number that a new group would take
			for checkInd in range(currInd+1, nVals):
				#go along the array sorting values into this group
				if uIDs[checkInd]==0 and norm(vals[currInd]-vals[checkInd])<tol:
					#this value is also not in a group yet and is a "duplicate" of the ungrouped element we are considering - they belong in the same group
					uIDs[checkInd] = uIDs[currInd]
			#we have made a new group of values and found all it's members... good job.
			currInd += 1

	#now uIDs is a grouping of all the values that we needed, and max(uIDs) is the number of unique values that we found
	unique = np.zeros((np.max(uIDs),), dtype=complex)
	for i in range(np.max(uIDs)):
		#average the values in group i and put them into the unique array
		unique[i] = np.mean( vals[uIDs==(i+1)] )
			
	return unique, uIDs

def UnitVector(i,n=3,dtype='complex'):
	'''
	Creates the cannonical (i+1)-th unit vector in R^n, as an array of complex (by default) data types
	INPUTS:
		i: 	int, python index corresponding to the (i+1)th component of the unit vector t be set to 1 - all others are 0
		n: 	(optional) int - default 3, size of the unit vector or dimension of R^n
		dtype: 	(optional) dtype - default complex, output array will have this dtype
	OUTPUTS:
		e 	: i-th cannonical unit vector in R^n as a numpy array
	'''
	
	e = np.zeros((n,), dtype=dtype)
	e[i] += 1
	
	return e

def CompToReal(z):
	'''
	Given a (n,) complex numpy array, return a (2n,) numpy array of floats containing the real and imaginary parts of the input complex vector. Real and imaginary parts are in subsequent entries, so real(z[i]) = x[2*i], imag(z[i]) = x[2*i+1]
	INPUTS:
		z: 	(n,) complex numpy array, array to be converted to (2n,) real array
	OUTPUTS:
		x: 	(2n,) numpy array, containing real and imaginary parts of the values in z according to real(z[k]) = x[2k], imag(z[k]) = x[2k+1]
	'''
	
	n = np.shape(z)[0]
	x = np.zeros((2*n,), dtype=float)
	for k in range(n):
		x[2*k] = np.real(z[k])
		x[2*k+1] = np.imag(z[k])
	
	return x

def RealToComp(x):
	'''
	Given a (2n,) numpy array, return a (n,) complex numpy array of numbers whose real and imaginary parts correspond index-neighbour pairs of the input array.
	INPUTS:
		x: 	(2n,) numpy array, array to be converted to (n,) complex array
	OUTPUTS:
		z: 	(n,) complex numpy array, containing complex numbers formed from the entries of z according to x[k] = z[2k]+i*z[2k+1]
	'''

	if np.shape(x)[0] % 2 !=0:
		raise ValueError('Cannot convert array of non-even length (%.1f) to complex array' % np.shape(x)[0])
	else:
		n = int(np.shape(x)[0]/2) #safe to use int() after the previous check
		z = np.zeros((n,), dtype=complex)
		for k in range(n):
			z[k] = x[2*k] + 1.j*x[2*k+1]

	return z

def PolyEval(x, cList):
	'''
	Given a value x and a list of coefficients, return the value of the polynomial expression cList[0]+x*cList[1]+x^2*cList[2]+...
	INPUTS:
		x 	: complex float, value to evaluate polynomial map at
		cList 	: list (to be cast to floats), polynomial coefficients
	OUTPUTS:
		pVal 	: polynomial evaluation at x
	'''
	pVal = 0.
	for i, coeff in enumerate(cList):
		if len(coeff)>0:
			pVal += float(coeff) * (x**i)
	
	return pVal

def NLII(f, df, v0, u, w0, maxItt=100, tol=1e-8, conLog=False):
	#do we need u... can't we just pick a random vector and hope or is this not OK? :L
	'''
	Solves the nonlinear eigenvalue problem f(w)v = 0 where (w,v) is an eigenpair in CxC^n, using the Nonlinear Inverse Iteration method (Guttel & Tisseur, 2017).
	INPUTS:
		f: 	lambda function, matrix-valued function of one argument, returning an (n,n) shape numpy array corresponding to f(w) as above.
		df: 	lambda function, matrix-valued function of one argument, returning an (n,n) shape numpy array corresponding to f'(w) element-wise.
		v0: 	(n,) numpy array, initial guess for for the eigenvector that the solver will use
		u: 	(n,) numpy array, used to normalise the output eigenvector and can also be used to avoid repeat convergence to the same eigenpair in the case of holomorphic f(w)
		w0: 	(complex) float, starting guess for the eigenvalue that the solver will use
		maxItt: 	(optional) int - default 100, maximum number of iterations to perform before giving up the search
		tol: 	(optional) float - default 1e-8, tolerance of solver
		conLog: 	(optional) bool - default False, if True then conIss is returned as a non-empty list containing log messages concerning failures of the solver
	OUTPUTS:
		wStar: 	complex float, the eigenvalue that was found
		vStar: 	(n,) complex numpy array, the eigenvector that was found
		conIss: 	list, empty if conLog is False. Otherwise conIss contains debugger information on cases when the solver failed (write more details Will).
	METHOD:
		tstk WHAT IS THIS WILL MAYBE YOU SHOULD WRITE THIS DOWN!!!!!! Use the paper
	'''
	
	#determine the dimension of the vectors that we are using
	n = np.shape(v0)[0]
	
	#Scipy cannot solve systems of equations with complex values, so we need a wrapper for this function which outputs real arrays. As such, the following function outputs a (2n,) vector of real values corresponding to the real and imaginary parts of the equation.

	return

#make this as general as possible! Possibly better to write an entirely new function and rename this one for the time being
# we should, for the time being, remove the graphs-specific stuff like theta, etc.
def Old_NLII(M, Mprime, v0, u, w0=np.pi, theta=np.zeros((2), dtype=float), maxItt=100, tol=1.0e-8, conLog=True, talkToMe=False):
	'''
	Solve the nonlinear eigenvalue problem M(w,theta)v = 0 for an eigenpair (w,v) using the Nonlinear Inverse Iteration method (see Guttel & Tisseur, 2017).
	INPUTS:
		M: 	lambda function, evaluates M(w,theta) at arguments (w,theta)
		Mprime: 	lambda function, evaluates d/dw M(w,theta) at arguments (w,theta)
		v0: 	(n,) numpy array, initial guess for the eigenvector
		u: 	(n,) numpy array, the vector u is used to normalise the output eigenvector and can be used to avoid repeat convergence to the same eigenpair in the case of holomorphic M
		w0: 	(optional) float, starting guess for the eigenvalue w. Default np.pi
		theta: 	(optional) (2,) numpy array, the quasimomentum value for this solve. Default [0,0]
		maxItt: 	(optional) int, maximum number of iterations to perform. Default 100
		tol: 	(optional) float, solution tolerance. Default 1.0e-8
		conLog: 	(optional) bool, if True then a list storing the information after each iteration plus any errors or warnings will be returned. Default True.
		talkToMe: 	(optional) bool, if True then the method will printout information about the converged solution at the end of the run. Default False
	OUTPUTS:
		wStar: 	complex float, eigenvalue that the solver converged to
		vStar: 	(n,) complex numpy array, eigenvector that the solver converged to
		conIss 	: (optional) list that logs any issues with non-convergence, only returned if conLog is True. More details below (write this Will)
	'''
	#first, figure out what size vectors we are dealing with!
	n = np.shape(u)[0]
	
	###GOT TO HERE - NEED TO MAKE THE SOLVING FUNCTION ENTIRELY REAL, SO TURN IT INTO A 2N VECTOR OF REAL VALUES, THEN CAST IT BACK AT THE END :l SEE https://stackoverflow.com/questions/21834459/finding-complex-roots-from-set-of-non-linear-equations-in-python OR SIMILAR
	
	#create lambda function that we will pass to fsolve in each loop of the iteration.
	fsolveFn = lambda x,w,v: np.matmul(M(w,theta),x) - np.matmul(Mprime(w,theta),v)
	#annoyingly, scipy cannot deal with complex valued equations, so we need a wrapper for this function which outputs real arrays. As such, the following function outputs a (2n,) vector of real values corresponding to the real and imaginary parts of the equation.
	def fRealFn(z,w,v):
		'''
		z should be a (2n,) numpy array which we convert to a complex-valued (n,) array, pass into fsolveFn, then cast the result back to a (2n,) real valued array.
		For internal use only, not to be seen externally.
		'''

		x = RealToComp(z) #cast back to complex
		fComplexValue = fsolveFn(x,w,v) #evaluate
		realOut = CompToReal(fComplexValue) #cast back to real array...
		
		return realOut
	
	#storage for iteration outputs
	wStore = np.zeros((maxItt+1), dtype=complex); wStore[0] = w0
	vStore = np.zeros((n,maxItt+1), dtype=complex); vStore[:,0] = v0	
	errStore = np.zeros((maxItt+1), dtype=float); errStore[0] = tol + 1 #auto-fail to start 1st iteration!
	#iteration counter
	currItt = 1
	
	while currItt<=maxItt and errStore[currItt-1]>tol:
		#whilst we have no exceeded the maximum number of iterations and the current tolerance in the solution is too high
		
		#solve M(w_k)x = M'(w_k)v_k for x; but we need to use z=real,imag (x) because SciPy
		func = lambda z: fRealFn(z,wStore[currItt-1],vStore[:,currItt-1])
		#for want of a better guess, use the current "eigenvector" as a guess of the solution...
		z = fsolve(func, CompToReal(vStore[:,currItt-1]))
		x = RealToComp(z) #cast back to complex array to save variables
		
		#set eigenvalue; w_{k+1} = w_k - u*v_k/u*x
		wStore[currItt] = wStore[currItt-1] - np.vdot(u,vStore[:,currItt-1])/np.vdot(u,x)
		
		#normalise eigenvalue
		vStore[:,currItt] = x/norm(x)
		
		#compute error, ||M(w_k+1)v_k+1||_2 should be small
		errStore[currItt] = norm(np.matmul(M(wStore[currItt],theta),vStore[:,currItt]))
		
		#incriment counter
		currItt += 1

	#dump values for manual investigation
	conIss = []
	#warnings that we might encounter
	if currItt>=maxItt:
		warn('Solver reached iteration limit')
		conIss.append('MaxItter')
	else:
		if talkToMe:
			print('Solver stopped at iteration %d' % (currItt-1))
		#shave off excess storage space to save some memory in this case
		#remember slicing in python is a:b goes from index a through to b-1 inclusive!
		wStore = wStore[0:currItt]
		vStore = vStore[:,0:currItt]
		errStore = errStore[0:currItt]
	if errStore[currItt-1]>tol:
		warn('Solver did not find a solution to the required tolerance: needed %.2e but only reached %.5e' % (tol, errStore[-1]))
		conIss.append('NoConv')
	else:
		if talkToMe:
			print('Difference from zero in converged eigenpair: %.5e' % errStore[currItt-1])
	if np.abs(norm(vStore[:,currItt-1])-1)>=tol:
		warn('Eigenvector is approximately 0')
		conIss.append('ZeroEVec')
	
	if conLog:
		#if we want the massive error log we need to return it here
		conIss.append(wStore)
		conIss.append(vStore)
		conIss.append(errStore)
		return wStore[currItt-1], vStore[:,currItt-1], conIss
	#otherwise, we just return the "answer"
	return wStore[currItt-1], vStore[:,currItt-1]