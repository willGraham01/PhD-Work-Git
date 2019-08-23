#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 10:24:00 2019

@author: will

This file contains test functions for the file GraphComponents.py and the methods it contains for solving for the sepctra of quantum graph problems. This is to split the two files apart so that we have one file that focuses on the actual functionality of the code, and another devoted solely to testing those functions.
"""

import csv
import numpy as np
from numpy import sin, cos, tan, real, imag
from warnings import warn
from numpy.linalg import norm
from GraphComponents_Master import Graph #import the actual Graph class
from GraphComponents_Master import SweepQM, RemoveDuplicates, FindSpectrum
from GraphComponents_Master import cot, cosec, UnitVector #star import is bad, but would work too if you don't mind the spyder warnings!
from GraphComponents_Master import NLII #non-linear inverse iteration solver
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

def TestVars():
	'''
	Creates some pre-set quantum graph instances for use in testing.
	Currently creates variables for:
		- TFR problem
	'''
	#filename for TFR variables
	TFR_file = 'TFR_Vars.csv'
	G_TFR = Graph(TFR_file)
	
	return G_TFR

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

def TFR_CheckEval(w, v, theta=np.zeros((2,), dtype=float), tol=1e-8, talkToMe=False):
	'''
	Checks whether the value that w converged to in the solver matches the analytic solution for the TFR problem.
	INPUTS:
		w 	: complex float, eigenvalue to test
		v 	: complex (n,) numpy array, eigenvector that pairs with the eigenvalue in question
		theta 	: (optional) (2,) numpy array, value of the quasimomentum parameter. Default [0.,0.]
		tol 	: (optional) tolerance for the checks, for best results set to the tolerance used in the solver. Default 1e-8
		talkToMe 	: (optional) bool, if true then the test program will print messages to the console as each check is run. Default False
	OUTPUTS:
		tf 	: bool, if true then the eigenvalue w solves the TFR to the accuracy of tol
		issues : list, contains all the flags that were raised when testing the eigenvalue
	'''
	tf = True #assume innocent until proven guilty
	issues = [] #log the issues that come up in a list
	G = TestVars()
	M_TFR = G.ConstructM()
	
	if not np.abs(imag(w))<tol:
		#bad start, the eigenvalue is not entirely real even by a computer's standards
		#give a warning about this
		issues.append('imagEval')
		if talkToMe: 
			warn('Eigenvalue has non-negligible imaginary part: %.5e' % imag(w))
	
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
	elif norm(np.matmul(M_TFR(w,theta),v))<tol:
		#so we still have an eigenvalue/vector pair - mst likely the eigenvalue is close to pi
		issues.append('multipleOfPi')
		if talkToMe:
			print('Eigenvalue is close to an odd multiple of pi')
	else:
		#the eigenvalue doesn't solve the TFR problem to our desired accuracy if we include the imaginary part
		if absRealDiff<tol:
			#but we do solve the TFR problem if we ignore the imaginary part...
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

def TFR_AutoChecker(nTrials=100, displayResult=True):
	'''
	Performs automated testing of the result of NLII in the case of the TFR problem. For each trial, random starting data and quasimomentum values are generated, and NLII is called to find a solution to the problem. We then validate the solution that is found using TFR_CheckEval.
	INPUTS:
		nTrials 	: (optional) int, nuumber of trials to perform in this run of testing. Defualt 100
		displayResult 	: (optional) bool, whether to display the results of the testing to the console at the end of the run. Default True
	OUTPUTS:
		badList 	: list, each member of the list is itself a list of the initial data, theta values, and supposed NLII solution that did not pass validation
		convList 	: list, each member logging a test case where the solver failed to converge correctly
		goodList 	: list, each member having the same structure as above, but for solutions that passed validation
	'''
	#build TFR functions that we need
	G = TestVars()
	M, Mprime = G.ConstructM(dervToo=True)
	
	record = np.ones((nTrials,), dtype=bool) #pass/fail record
	badList = [] #record of bad trials so that we can check what's happening
	goodList = [] #record of all the trials that passed, in case we want them
	convList = [] #record all the failed convergences
	
	#generate starting data and quasimomentum for the solver, for each trial
	w0Samples = np.random.uniform(0, 2*np.pi, (nTrials,))
	v0Samples = np.random.uniform(0, 1, (3,nTrials)) + 0.j #complex array for vector samples
	thetaSamples = np.random.uniform(-np.pi, np.pi, (2,nTrials)) #quasimomentum samples
	u = UnitVector(0) #vector u that has to be passed
	
	print('Beginning trials...')
	for i in range(nTrials):
		v0Samples[:,i] = v0Samples[:,i]/norm(v0Samples[:,i]) #normalise vector v0
		
		#perform NLII to find the solution to this problem
		w, v, conIss = NLII(M, Mprime, v0Samples[:,i], u, w0=w0Samples[i], theta=thetaSamples[:,i], maxItt=1000)
		
		if len(conIss)<=3: #this is shorter or equal to the usual information dump that NLII provides
			#there were no issues with the convergence in the numerical method, so we should have an eigenvalue
			#now validate the solution with our analytic solution
			tf, record = TFR_CheckEval(w, v, theta=thetaSamples[:,i], talkToMe=False)
			if tf:
				#if we passed, then we found a genuine solution to the problem... yay :)
				goodList.append([w0Samples[i], v0Samples[:,i], thetaSamples[:,i], w, v])
			else:
				#this time the method failed for some reason... look into the problems and return the result of the check
				badList.append([record, w0Samples[i], v0Samples[:,i], thetaSamples[:,i], w, v])
				print('Trial', i, 'failed, record logs:', record)
		else:
			convList.append([conIss[0:-3], w0Samples[i], v0Samples[:,i], thetaSamples[:,i], w, v])
	
	if displayResult:
		#show the result of the test with a print to the console
		print('Performed %d trials' % (nTrials))
		print('Passes: %d, Fails (convergence): %d, Fails (evaluation): %d' % (len(goodList),len(convList),len(badList)))
	
	return badList, convList, goodList

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

def ExtractToCSV(G, TheoryEvals, tSpace=np.linspace(-np.pi,np.pi,num=250), fName='results', sym=False, nSamples=7, wRange=np.asarray([0,2*np.pi])):
	'''
	For the graph G, sweep through the QM values in tSpace and save the eigenvalues that are found to a csv file. Repeat for each value in tSpace, adding the e'vals for each value on a new line of the csv file. For use in creating an animation that displays the numerical results.
	INPUTS:
		G 	: Graph instance, problem to solve
		TheoryEvals 	: function, taking inputs theta and wRange and returning the exact eigenvalues for the problem corresponding to G in the range wRange at the QM value in theta.
		tSpace 	: (optional) (n,) numpy array, values of QM to sweep through. Default np.linspace(-pi,pi,num=250)
		fName 	: (optional) str, file names for the outputs will have this string appended to them. Default 'results'
		sym 	: (optional) bool, if True then we assume the QM is symmetric so we only do one loop over the QM, with theta2=0. Default False
		nSamples 	: (optional) int, number of initial guesses to give the NLII solver. Default 7
		wRange 	: (optional) (2,) numpy array, range to search for eigenvalues. Default [0,2pi]
	OUTPUTS:
		compValList 	: list of numpy arrays, each array is the set of eigenvalues found computationally for each theta value
		exValList 	: list of numpy arrays, as above but the exact eigenvalues
	'''
	#find computational eigenvalues
	fNameComputational = fName + 'Comp.csv'
	#compValList = []
	fComp = open(fNameComputational,'a')	
	theta = np.zeros((2,), dtype=float)
	print('Finding computational eigenvalues')
	for t1 in tSpace:
		print('theta_1:',t1)
		theta[0] = t1
		if not sym:
			for t2 in tSpace:
				theta[1] = t2
				eVals, _ = SweepQM(G, nSamples, theta, wRange)
				#truncate so that only the "in-range" eigenvalues are found
				inRangeVals = (eVals < wRange[1]) & (eVals > wRange[0])
				eVals = eVals[inRangeVals]
				#compValList.append(np.hstack((theta, eVals)))
				np.savetxt(fComp, [np.hstack((theta, eVals))], delimiter=',')
		else:
			#symmetric case
			#find eigenvalues
			eVals, _ = SweepQM(G, nSamples, theta, wRange)
			#truncate so that only the "in-range" eigenvalues are found
			inRangeVals = (eVals < wRange[1]) & (eVals > wRange[0])
			eVals = eVals[inRangeVals]
			#compValList.append(np.hstack((theta, eVals)))
			np.savetxt(fComp, [np.hstack((theta, eVals))], delimiter=',')
	
	##need to pad out the computational arrays with extra zeros or something?
#	for row in compValList:
#		np.savetxt(fComp, [row], delimiter=',')
	fComp.close()
	
	#find theoretical eigenvalues
	fNameExact = fName + 'Exact.csv'
	#exValList = []
	fExact = open(fNameExact,'a')
	theta = np.zeros((2,), dtype=float)
	print('Finding exact eigenvalues')
	for t1 in tSpace:
		print('theta_1',t1)
		theta[0] = t1
		if not sym:
			for t2 in tSpace:
				theta[1] = t2
				eVals = TheoryEvals(theta, wRange)
				#exValList.append(np.hstack((theta, eVals)))
				np.savetxt(fExact, [np.hstack((theta, eVals))], delimiter=',')
		else:
			#symmetric case
			#find the theoretical eigenvalues in this case
			eVals = TheoryEvals(theta, wRange)
			#exValList.append(np.hstack((theta, eVals)))
			np.savetxt(fExact, [np.hstack((theta, eVals))], delimiter=',')
#	for row in exValList:
#		np.savetxt(fExact, [row], delimiter=',')
	fExact.close()
	
	return fNameComputational, fNameExact#compValList, exValList

def TFR_ExactEvals(theta, wRange=np.asarray([0,2*np.pi])):
	'''
	For the TFR problem (without coupling constants) returns the exact eigenvalues corresponding to the QM being theta, in the range wRange.
	INPUTS:
		theta 	: (2,) numpy array, value of the QM
		wRange 	: (optional) (2,) numpy array, range to return eigenvalues in. Default [0,2pi]
	OUTPUTS:
		eVals 	: numpy array, all eigenvalues in the range wRange
	'''
	valList = []
	#eigenvalues solve cos(w) = cos((t1-t2)/2)cos((t1+t2)/2)
	RHS = cos((theta[0]-theta[1])/2)*cos((theta[0]+theta[1])/2)
	#obtain primary root in the range [0,pi]
	primeVal = np.arccos(RHS)
	#obtain secondary root in the range [pi,2pi]
	secVal = 2*np.pi - primeVal
	
	#now we just need to find the repeated roots for the e'vals in the event that wRange is not [0,2pi]
	if primeVal<=wRange[0]:
		#primary root is too low for the range we are interested in
		transPVal = primeVal
		while transPVal<=wRange[0]:
			#whilst we are lower than the lower limit of wRange
			transPVal += 2*np.pi #add on periods to reach the lower limit
		while transPVal<=wRange[1]:
			#we are now above the lower limit. All the while we are below the upper limit, we are interested in this as a root
			valList.append(transPVal)
			transPVal += 2*np.pi
	elif primeVal>=wRange[1]:
		#we started above the upper bound
		transPVal = primeVal
		while transPVal>=wRange[1]:
			transPVal -= 2*np.pi
		while transPVal>=wRange[0]:
			valList.append(transPVal)
			transPVal -= 2*np.pi
	else:
		#we started off with wRange[0]<primeVal<wRange[1] - so we need to search both ways
		transPVal = primeVal
		#search "right"
		while transPVal<=wRange[1]:
			valList.append(transPVal)
			transPVal += 2*np.pi
		transPVal = primeVal - 2*np.pi
		#"search left"
		while transPVal>=wRange[0]:
			valList.append(transPVal)
			transPVal -= 2*np.pi
	#now we do all that again, but for the secondary root!
	if secVal<=wRange[0]:
		#primary root is too low for the range we are interested in
		transSVal = secVal
		while transSVal<=wRange[0]:
			#whilst we are lower than the lower limit of wRange
			transSVal += 2*np.pi #add on periods to reach the lower limit
		while transSVal<=wRange[1]:
			#we are now above the lower limit. All the while we are below the upper limit, we are interested in this as a root
			valList.append(transSVal)
			transSVal += 2*np.pi
	elif secVal>=wRange[1]:
		#we started above the upper bound
		transSVal = secVal
		while transSVal>=wRange[1]:
			transSVal -= 2*np.pi
		while transSVal>=wRange[0]:
			valList.append(transSVal)
			transSVal -= 2*np.pi
	else:
		#we started off with wRange[0]<primeVal<wRange[1] - so we need to search both ways
		transSVal = secVal
		#search "right"
		while transSVal<wRange[1]:
			valList.append(transSVal)
			transSVal += 2*np.pi
		transSVal = secVal - 2*np.pi
		#"search left"
		while transSVal>wRange[0]:
			valList.append(transSVal)
			transSVal -= 2*np.pi
	#form eigenvalues into numpy array
	eVals = np.asarray(valList)
	return eVals

def ResultsToPlot(fName):
	'''
	Inpterpret the results file given and produce the dispersion plot(s) for this data
	INPUTS:
		fName 	: string, file name and path of the .csv file holding the data
	'''
	#read input file line-by-line
	lines = []
	with open(fName, "r") as f:
		reader = csv.reader(f, delimiter=",")
		for line in reader:
			lines.append(line)
	for i in range(0,len(lines)):
		#first remove any whitespace and brackets in the strings that might have occured when they saved
		for j in range(len(lines[i])):
			lines[i][j] = lines[i][j].replace("(","")
			lines[i][j] = lines[i][j].replace(")","")
			lines[i][j] = lines[i][j].replace(" ","")
			lines[i][j] = complex(lines[i][j])
	#reconstruct tSpace variable, and determine how many values have been found per QM value
	tSpace = []
	lineLen = len(lines[0])
	tSpace.append(np.real(lines[0][0]))
	for i in range(1,len(lines)):
		#remember that we cycle through theta_2 then move on in theta_1 so we only want the unique theta_1 values!
		if lines[i-1][0]!=lines[i][0]:
			tSpace.append(np.real(lines[i][0]))
		if len(lines[i])>lineLen:
			lineLen = len(lines[i])
	tSpace = np.asarray(tSpace)
	#reconstruct the matrix of values. This is a tSpace*tSpace*maxLen array, the 3rd dimension corresponds to the branch number that we are examining
	tLen = len(tSpace)
	maxEvalsOnLine = lineLen-2
	eValData = np.zeros((tLen,tLen,maxEvalsOnLine), dtype=complex)
	for i in range(tLen):
		for j in range(tLen):
			for k in range(2,len(lines[i*tLen + j])):
				eValData[i,j,k-2] = lines[i*tLen + j][k]
	#this will give us a matrix of the values that we found, but there is still the issue of mismatching lengths and hence branches being "interwined"
	
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	tSpaceX, tSpaceY = np.meshgrid(tSpace, tSpace)
	# Plot the surface.
	surf = ax.plot_surface(tSpaceX, tSpaceY, np.real(eValData[:,:,0]), cmap=cm.coolwarm,linewidth=0, antialiased=False)
	# Customize the z axis.
	ax.set_zlim(np.min(eValData[:,:,0]), np.max(eValData[:,:,0]))
	ax.set_zlabel('$\omega$')
	ax.set_xlabel('$\theta_1$')
	ax.set_ylabel('$\theta_2$')
	# Add a color bar which maps values to colors.
	fig.colorbar(surf, shrink=0.5, aspect=5)
	plt.show()

	return eValData, tSpace

def QuickFig(tSpace):
	
	tX, tY = np.meshgrid(tSpace, tSpace)
	Z = 2*(np.cos(tX/2)**2)/3 + 1*(np.cos(tY/2)**2)/3
	Z = 2*np.arccos(np.sqrt(Z))
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	surf = ax.plot_surface(tX, tY, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
	fig.colorbar(surf, shrink=0.5, aspect=5)
	plt.show()
	
	return Z

#delete once complete, but for now just auto give me the test variables
#G_TFR, M_TFR, vTFR, aTFR, G_EKK, M_EKK, vEKK, aEKK = TestVars()
G_TFR = TestVars()
M_TFR, Mprime_TFR = G_TFR.ConstructM(dervToo=True)