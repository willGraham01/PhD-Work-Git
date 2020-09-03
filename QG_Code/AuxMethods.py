#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 18:29:32 2020

@author: will

This file contains various functions that I may wish to employ across multiple files, scripts and potentially integrate into classes for my PhD.

Rough table of contents:
	- csc
	- cot
	- Spiral
	- AbsLess
	- RemoveDuplicates
	- UnitVector
	- CompToReal
	- RealToComp
	- PolyEval
	- NLII
	- EvalCheck
"""

from warnings import warn

import numpy as np
from numpy import sin, tan
from numpy.linalg import norm

from scipy.linalg import solve, lstsq, LinAlgError

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

def Spiral(n, npArray=True):
	'''
	Give a number n in N_0, provides the co-ordinate in Z^2 corresponding to the n^th co-ordinate in the spiral bijection from N --> Z^2. See https://math.stackexchange.com/questions/163080/on-a-two-dimensional-grid-is-there-a-formula-i-can-use-to-spiral-coordinates-in for details on where the formula was obtained.
	INPUTS:
		n: 	int, natural number or 0 to be mapped to co-ordinate
		npArray: 	(optional) bool - default True, if True then co-ordinate will be returned as a numpy array, otherwise as a list
	OUTPUTS:
		spiralCoord: 	(2,) int numpy array or list, co-ordinate in Z^2 this element is mapped to
	'''
	
	m = np.floor(np.sqrt(n))
	if m%2==1: #if m odd
		k = (m - 1) / 2
	elif n >= m*(m + 1): #if m even and n>= m(m+1)
		k = m/2
	else:
		k = m/2 - 1
	
	if 2*k*(2*k + 1) <= n and n <= (2*k + 1)*(2*k + 1):
		spiralCoord = [n - 4*k*k - 3*k, k]
	elif (2*k + 1)*(2*k + 1) < n and n <= 2*(k + 1)*(2*k + 1):
		spiralCoord = [k + 1, 4*k*k + 5*k + 1 - n]
	elif 2*(k + 1)*(2*k + 1) < n and n <= 4*(k + 1)*(k + 1):
		spiralCoord = [4*k*k + 7*k + 3 - n, -k - 1]
	elif 4*(k + 1)*(k + 1) < n and n <= 2*(k + 1)*(2*k + 3):
		spiralCoord = [-k - 1, n - 4*k*k - 9*k - 5]
	else:
		raise ValueError('No spiral co-ordinate found for n=%d: m=%d (sqrt(n)=%.2f), k=%d \n' % (n, m, np.sqrt(n), k))
	
	if npArray:
		return np.asarray(spiralCoord, dtype=int)
	else:
		return spiralCoord

def RemoveListDuplicates(l):
	'''
	Given a list, return a list that contains only the unique entries in the list
	INPUTS:
		l: 	list, list to have duplicates removed
	OUTPUTS:
		unique: 	list, lst of unique items
	'''

	unique = list(dict.fromkeys(l))
	return unique

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
		raise ValueError('Cannot convert array of non-even length (%d) to complex array' % np.shape(x)[0])
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

def NLII(f, df, v0, u, w0, maxItt=100, tol=1e-8, conLog=False, retNItt=False, talkative=False):
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
		retNItt: (optional) bool - default False, if True then the number of iterations that were performed is returned as a 4th output argument
		talkative: 	(optional) bool - defaul False, if True then the solver will provide updates to the console concerning the latest eigenvalue and eigenvector estimates, on each iteration
	OUTPUTS:
		wStar: 	complex float, the eigenvalue that was found
		vStar: 	(n,) complex numpy array, the eigenvector that was found
		conIss: 	list, empty if conLog is False. Otherwise conIss contains debugger information on cases when the solver failed (write more details Will)
		currItt: (optional) int, if retNItt is True then this is the number of iterations that the solver performed.
	METHOD:
		Choose an initial pair (w_0, v_0) with ||v_0||=1 and non-zero vector u that will be used to check normality of the solution.
		Then until convergence or escape do:
			Solve f(w_k)v_k+1 = df(w_k)v_k for v_k+1
			Set w_k+1 = w_k - u.v_k/u.v_k+1
			Normalise v_k+1
	'''
	
	#determine the dimension of the vectors that we are using
	n = np.shape(v0)[0]
	#storage for iteration outputs
	wStore = np.zeros((maxItt+1), dtype=complex); wStore[0] = w0
	vStore = np.zeros((n,maxItt+1), dtype=complex); vStore[:,0] = v0	
	errStore = np.zeros((maxItt+1), dtype=float); errStore[0] = tol + 1 #auto-fail to start 1st iteration!
	#iteration counter
	currItt = 1
	#dump values for manual investigation
	conIss = []
	#now we start the iteration
	if talkative:
		print('Initial data: eigenvalue: %.5e + %.5e i' %(np.real(w0), np.imag(w0)))
		print('Eigenvector starting guess:')
		print(v0)
	while currItt<=maxItt and errStore[currItt-1]>tol:
		#whilst we have not exceeded the maximum number of iterations and the current error in the solution is too high
		#solve f(w_k)x = df(w_k)v_k for x
		RHS = np.matmul( df(wStore[currItt-1]), vStore[:,currItt-1])
		LHSMat = f(wStore[currItt-1])
		#We might have a singular or near-singular LHSMat, the Pythonic way to handle this is to assume it's not singular and then beg forgiveness if it is
		try:
			#use scipy.linalg's solve function to solve the linear system, but this will fail for singular matrices!
			x = solve(LHSMat, RHS)
		except LinAlgError:
			#if we get this error, LHSMat is singular. Try a least squares fit instead
			x, _, _, _ = lstsq(LHSMat, RHS)
			#if sucessful, then place this issue in conIss either way!
			if 'lstsqReq' not in conIss:
				conIss.append('lstsqReq')
			if talkative:
				print('Least squares regression required')
		#set eigenvalue; w_{k+1} = w_k - u*v_k/u*x
		wStore[currItt] = wStore[currItt-1] - np.vdot(vStore[:,currItt-1],u)/np.vdot(x,u)
		#normalise eigenvalue
		vStore[:,currItt] = x/norm(x)
		#compute error, ||f(w_k+1)v_k+1||_2 should be small
		errStore[currItt] = norm(np.matmul(f(wStore[currItt]),vStore[:,currItt]))
		if talkative:
			#print info to screen
			print('Interation %d, eigenvalue: %.5e + %.5e i' %(currItt, np.real(wStore[currItt]), np.imag(wStore[currItt])))
			print('Eigenvector approximation:')
			print(vStore[:,currItt])
		#incriment counter
		currItt += 1
	#warnings that we might encounter
	#max iteration limit
	if currItt>=maxItt:
		warn('Solver reached iteration limit')
		conIss.append('MaxItter')
	else:
		#if we didn't hit max iterations we can shave off the extra memory space we allocated for our variable stores
		#remember slicing in python is a:b goes from index a through to b-1 inclusive!
		wStore = wStore[0:currItt]
		vStore = vStore[:,0:currItt]
		errStore = errStore[0:currItt]
	#tolerance not reached
	if errStore[currItt-1]>tol:
		warn('Solver did not find a solution to the required tolerance: needed %.2e but only reached %.5e' % (tol, errStore[-1]))
		conIss.append('NoConv')
	#converged to zero eigenvector
	if np.abs(norm(vStore[:,currItt-1])-1)>=tol:
		warn('Eigenvector is approximately 0')
		conIss.append('ZeroEVec')
	#rappend to the log if we wanted this information, else we return an empty list
	if conLog:
		conIss.append(wStore)
		conIss.append(vStore)
		conIss.append(errStore)
	#otherwise, we just return the "answer"
	
	if retNItt:
		return wStore[currItt-1], vStore[:,currItt-1], conIss, currItt-1
	else:
		return wStore[currItt-1], vStore[:,currItt-1], conIss
	
def EvalCheck(M, w, v, theta=np.zeros((2), dtype=float)):
	'''
	Given the M-matrix, the eigenvalue, and eigenvector v, determine the norm of M(w)v.
	This should be 0 if (w,v) is a solution pair to the generalised eigenvalue problem.
	INPUTS:
		M: 	lambda function, matrix-valued function of one argument, returning an (n,n) shape numpy array corresponding to M(w,theta).
		w: 	float, generalised eigenvalue to be checked
		v: 	(n) complex float, generalised eigenvector to be checked
		theta: 	(optional) (2,) float numpy array - default [0,0], quasimomentum value at which the eigenpair was found
	OUTPUTS:
		tf: 	bool, if True then the pair (w,v) is an eigenpair at the input value of theta
		RHSVec: 	(n,) complex numpy array, the result of M(w,theta)v
	'''
	
	RHSVec = np.matmul(M(w, theta),v) #this should be the zero vector if NLII converged
	if norm(RHSVec) <= 1e-8:
		tf = True
	else:
		tf = False
		warn('Tested pair does not produce zero vector upon product with matrix')
	
	return tf, RHSVec

#delete this Will when you are done with testing the solver
if __name__=='__main__':
	
	#A = np.asarray([[1,0,0],[0,2,0],[0,0,3]], dtype=float)
	A = np.random.rand(3,3)
	I = np.eye(3)
	def f(w):
		return A - (w**2)*I - w*I
	def df(w):
		return -2*w*I - I