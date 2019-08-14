#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 10:24:00 2019

@author: will

This file contains test functions for the file GraphComponents.py and the methods it contains for solving for the sepctra of quantum graph problems. This is to split the two files apart so that we have one file that focuses on the actual functionality of the code, and another devoted solely to testing those functions.
"""

import numpy as np
from numpy import sin, cos, tan, real, imag
from warnings import warn
from numpy.linalg import norm
from GraphComponents import Graph #import the actual Graph class
from GraphComponents import cot, cosec #star import is bad, but would work too if you don't mind the spyder warnings!
from GraphComponents import NLII #non-linear inverse iteration solver

def TestVars():
	'''
	Creates some pre-set quantum graph instances for use in testing.
	Currently creates variables for:
		- TFR problem
	'''
	#Test variables for the TFR problem
	posTFR = np.asarray( [ [0.5, 1.0, 0.5], [1.0, 0.5, 0.5] ] )
	adjMatTFR = np.asarray( [ [0, 0, 1], [0, 0, 1], [1, 1, 0] ] )
	#when building this, remember that python indexes from 0 not 1
	theMatTFR = np.zeros((3,3,2))
	theMatTFR[0,2,:] = np.asarray([0.0,1.0]) #v1 is vertically above v3... so theta_2 should be here
	theMatTFR[2,0,:] = np.asarray([0.0,1.0]) #symmetry
	theMatTFR[1,2,:] = np.asarray([1.0,0.0]) #v2 is right of v3
	theMatTFR[2,1,:] = np.asarray([1.0,0.0]) #symmetry
		
	G_TFR = Graph(posTFR, adjMatTFR, theMatTFR)
	M_TFR = G_TFR.ConstructM()
	
	#Test variables for the problem in the EKK paper
#	posEKK = np.asarray( [ [], [] ] )
#	adjMatEKK = np.asarray( [ [], [], [], [] ] )
#	theMatEKK = np.zeros((4,4,2))
#	#THEMAT STUFF HERE
#	G_EKK = Graph(posEKK, adjMatEKK, theMatEKK)
#	M_EKK = G_EKK.ConstructM()
	
	return G_TFR, M_TFR, posTFR, adjMatTFR#, G_EKK, M_EKK, posEKK, adjMatEKK

##TFR checking functions
def TFR_Exact(w, theta=np.zeros((2))):
	'''
	Exact M-matrix for the TFR problem
	INPUTS:
		w 	: float, value of w
		theta 	: (optional) (2,) numpy array, value of the quasimomentum parameters
	OUTPUTS:
		mat 	: M-matrix for the TFR problem, evaluated at (w,theta)
	'''
	
	mat = np.zeros((3,3), dtype=complex)
	mat[0,0] = -2*w*cot(w/2)
	mat[1,1] = -2*w*cot(w/2)
	mat[2,2] = -4*w*cot(w/2)
	mat[0,2] = 2*w*np.cos(theta[1]/2)*cosec(w/2)
	mat[2,0] = 2*w*np.cos(theta[1]/2)*cosec(w/2)
	mat[1,2] = 2*w*np.cos(theta[0]/2)*cosec(w/2)
	mat[2,1] = 2*w*np.cos(theta[0]/2)*cosec(w/2)
	
	return mat

def TFR_CheckEval(w, theta=np.zeros((2,), dtype=float), tol=1e-8, talkToMe=False):
	'''
	Checks whether the value that w converged to in the solver matches the analytic solution for the TFR problem.
	INPUTS:
		w 	: complex float, eigenvalue to test
		theta 	: (optional) (2,) numpy array, value of the quasimomentum parameter. Default [0.,0.]
		tol 	: (optional) tolerance for the checks, for best results set to the tolerance used in the solver. Default 1e-8
		talkToMe 	: (optional) bool, if true then the test program will print messages to the console as each check is run. Default False
	OUTPUTS:
		tf 	: bool, if true then the eigenvalue w solves the TFR to the accuracy of tol
		issues : list, contains all the flags that were raised when testing the eigenvalue
	'''
	tf = True #assume innocent until proven guilty
	issues = [] #log the issues that come up in a list
	
	if not np.abs(imag(w))<tol:
		#bad start, the eigenvalue is not entirely real even by a computer's standards
		#give a warning about this
		issues.append('imagEval')
		if talkToMe: 
			warn('Eigenvalue has non-negligible imaginary part: %.5e' % imag(w))
		#flag the issue in the return
		tf = False
	
	#now the actual TFR check, we know that eigenvalues of the M-matrix fall precisely at those values for which
	# cos(w) = cos((theta[0]+theta[1])/2)cos((theta[0]-theta[1])/2)
	cw = cos(w)
	cwRealOnly = cos(real(w))
	cTheory = cos((theta[0]+theta[1])/2)*cos((theta[0]-theta[1])/2)
	absDiff = np.abs(cw-cTheory)
	absRealDiff = np.abs(cwRealOnly-cTheory)
	if absDiff<tol:
		#the eigenvalue has converged pretty well even if we leave the imaginary part in
		if absRealDiff<tol:
			#the eigenvalue has also converged pretty well even if we remove the imaginary part - everything is good
			pass
		else:
			#good convergence of the imaginary eigenvalue, but removing the imaginary part causes issues with convergence. As such, it's likely that we also failed to be essentially real
			tf = False
			issues.append('imagPartNeeded')
			if talkToMe: 
				warn('Eigenvalue does not solve TFR problem unless imaginary part is included')
	else:
		#the eigenvalue doesn't solve the TFR problem to our desired accuracy if we include the imaginary part
		if absRealDiff<tol:
			#but we do solve the TFR problem if we ignore the imaginary part...
			tf = True
			issues.append('imagPartRemoved')
			if talkToMe: 
				warn('Eigenvalue only solves TFR problem if imaginary part removed')
		else:
			#this eigenvalue is just rubbish!
			tf = False
			issues.append('notEVal')
			if talkToMe: 
				warn('Value given is likely not an eigenvalue')
		
	return tf, issues

def TFR_CheckEvec(v, theta=np.zeros((2,), dtype=float), tol=1e-8, talkToMe=False):
	'''
	Checks whether the value that w converged to in the solver matches the analytic solution for the TFR problem.
	INPUTS:
		v 	: (n,) complex numpy array, eigenvector to test
		theta 	: (optional) (2,) numpy array, value of the quasimomentum parameter. Default [0.,0.]
		tol 	: (optional) tolerance for the checks, for best results set to the tolerance used in the solver. Default 1e-8
		talkToMe 	: (optional) bool, if true then the test program will print messages to the console as each check is run. Default False
	OUTPUTS:
		tf 	: bool, if true then the eigenvector v solves the TFR to the accuracy of tol
		issues : list, contains all the flags that were raised when testing the eigenvalue
	'''
	tf = True #innocent until guilty
	issues = []
	
	return tf, issues

def CompareConstructions(exact, computational, nSamples=1000, theta1Present=True, theta2Present=False, wRange=np.asarray([-np.pi,np.pi]), thetaRange=np.asarray([-np.pi,np.pi]), tol=1e-8):
	'''
	Given two function handles, exact and computational, performs nSamples evaluations of both functions over prescribed intervals to determine if the computationally constructed solution and the exact form of the M-matrix agree.
	INPUTS:
		exact 	: lambda function, given (w,theta) produces the M-matrix from an analytic expression
		computational 	: lambda function, given (w,theta) produces the M-matrix from computational construction
		nSamples 	: (optional) int, number of samples in each interval - total samples is nSamples^3
		theta1Present 	: (optional) bool, if true then we will sample theta1 values too - if false then we assume theta1=0. Default True
		theta2Present 	: (optional) bool, if true then we will sample theta2 values too - if false then we assume theta2=0. Default False
		wRange 	: (optional) (2,) numpy array, interval of w to test. Default [-pi, pi]
		thetaRange 	: (optional) (2,) numpy array, interval of theta to test. Default [-pi,pi]
		tol 	: (optional) float, tolerance of Frobenius norm to accecpt as numerical error
	OUTPUTS:
		failList 	: list, each entry in the list is a further list of 2 members: the w and theta values of the failures for the exact solution to match the computational one
	'''
	#sampled calues of w
	wVals = np.linspace(wRange[0],wRange[1],num=nSamples,endpoint=True)
	
	#sampled values of theta
	tVals = np.zeros((2,nSamples))
	if theta1Present:
		tVals[0,:] = np.linspace(thetaRange[0],thetaRange[1],num=nSamples,endpoint=True)
	if theta2Present:
		tVals[1,:] = np.linspace(thetaRange[0],thetaRange[1],num=nSamples,endpoint=True)
	
	matches = 0
	fails = 0
	failList = []
	nTrials = nSamples
	
	for i in range(nSamples):
		if (theta1Present and theta2Present):
			nTrials = nSamples*nSamples*nSamples
			for j in range(nSamples):
				for k in range(nSamples):
					e = exact(wVals[i], tVals[[0,1],[j,k]])
					c = computational(wVals[i], tVals[[0,1],[j,k]])
			
					#Frobenius norm
					froNorm = norm(e-c,'fro')		
					#determine mismatch cases
					if froNorm>=tol:
						#too much mismatch
						fails += 1
						failList.append([wVals[i], tVals[[0,1],[j,k]]]) #append the fail points
					else:
						#pass test
						matches += 1
		elif theta1Present or theta2Present:
			#only looping over theta1 values
			nTrials = nSamples*nSamples
			for j in range(nSamples):
				e = exact(wVals[i], tVals[:,j])
				c = computational(wVals[i], tVals[:,j])
		
				#Frobenius norm
				froNorm = norm(e-c,'fro')		
				#determine mismatch cases
				if froNorm>=tol:
					#too much mismatch
					fails += 1
					failList.append([wVals[i], tVals[:,j]]) #append the fail points
				else:
					#pass test
					matches += 1
		else:
			#only looping over w values
			e = exact(wVals[i], tVals[:,j])
			c = computational(wVals[i], tVals[:,j])
	
			#Frobenius norm
			froNorm = norm(e-c,'fro')		
			#determine mismatch cases
			if froNorm>=tol:
				#too much mismatch
				fails += 1
				failList.append([wVals[i], tVals[:,j]]) #append the fail points
			else:
				#pass test
				matches += 1		
						
	print('Test result:')
	print('Passes: %d ( %.3f )' % (matches, float(matches)/nTrials))
	print('Fails: %d ( %.3f )' % (fails, float(fails)/nTrials))
	print('Returning list of failure cases')
	
	return failList

def UnitVector(i,n=3):
	'''
	Creates the cannonical (i+1)-th unit vector in R^n, albeit as an array of complex data types
	INPUTS:
		i 	: int, python index corresponding to the (i+1)th component of the unit vector t be set to 1 - all others are 0
		n 	: (optional) int, size of the unit vector or dimension of R^n. Default 3
	OUTPUTS:
		e 	: i-th cannonical unit vector in R^n as an array of complex data types
	'''
	e = np.zeros((n,), dtype=complex)
	e[i] = 1.+0.j
	
	return e

#delete once complete, but for now just auto give me the test variables
#G_TFR, M_TFR, vTFR, aTFR, G_EKK, M_EKK, vEKK, aEKK = TestVars()
G_TFR = TestVars()[0] #outputs are returned in a tuple, of which G_TFR is the first
M_TFR, Mprime_TFR = G_TFR.ConstructM(dervToo=True)
u = UnitVector(0,3)
v0 = UnitVector(0,3)
theta = np.zeros((2,), dtype=float)