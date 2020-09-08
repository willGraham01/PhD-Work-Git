#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 08:59:59 2020

@author: will

This file is designed to probe the stability conjecture for our M-Matrix eigenvalue problem.
Our goal is to solve the following generalised eigenvalue problem:
	M(w; theta)v = 0.
	
We do this by the NLII solver, for each value of the quasimomentum (theta) needing to be considered, and returning corresponding values of w.
The question we ask is thus; how does the converged solution (w, v) change as theta is changed? Is there a stable relation is some sense (IE small changes in theta implies small changes in w and v)?
In this file we look to address this issue.
"""

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

from warnings import warn

from AuxMethods import NLII, UnitVector, Spiral
from Graph_TestFns import TFR_Setup

def CreateHandle(f, thetaVal):
	'''
	Given a function f of two variables, one of which is the vector theta, return a lambda function of one variable which is simply f( . , thetaVal)
	INPUTS:
		f: function f(x,theta), the argument theta should be an (n) float numpy array
		thetaVal: (n) float numpy array, value of theta that should be fixed in f
	OUTPUTS:
		fFixed: lambda function fFixed(x), acts as fFixed(x) = f(x,thetaVal)
	'''
	
	fFixed = lambda x: f(x,theta=thetaVal)
	
	return fFixed

def WriteRow(fName, qmData, runNumber, eVal, eVec, conIss):
	'''
	Open the file fName, append the data to the file along a row, then close the file. This prevents premature termination of a script from dumping any of it's output to a file.
	INPUTS:
		fName: 	str, filename of .csv file to append to
		qmData: 	(2,) numpy array, quasimomentum
		runNumber: 	int, run number from which initial data was taken
		eVal: 	float, eigenvalue that was found for this run
		eVec: 	(n,) numpy array, eigenvector that was found
		conIss: 	list of strings, warning flags obtained from running NLII solver
	'''
	
	theList = [qmData[0], qmData[1], runNumber, eVal]
	formats = ['%e', '%e', '%d', '%e']
	for vi in eVec:
		theList.append(vi)
		formats.append('%e')
	if conIss: #empty list is implicitly false, so if conIss is non-empty, an error occurred
		theList.append(1)
	else:
		theList.append(0)
	formats.append('%d')
	
	saveArray = np.asarray(theList)
	
	f = open(fName, 'a')
	#np.savetxt(f, [saveArray], delimiter=',', fmt=formats)
	np.savetxt(f, [saveArray], delimiter=',')
	f.close()
	return

def SolveStepSolve(M, dM, vInit, u, wInit, t1End = np.pi, t2End = np.pi, tPts = 1000, fileName=False, **kwargs):
	'''
	The purpose of this function is, given some initial data for the NLII solver, to attempt to solve the NLII for other values of theta using the previous answer that was found as the starting guess for the next value of theta. The program assumes that the starting value of theta (for the first solve) is the zero vector, and the interval for each value of theta is symmetric about 0, for ease when stepping through theta and performing the remaining solves
	INPUTS:
		M: 	function, M(w,theta) is the value of the M-matrix function at the value w and for the quasimomentum theta provided
		dM: 	function, dM(w,theta) is the value of the function M'(w, theta)
		vInit: 	(n,) numpy array, initial guess for for the eigenvector that the solver will use for the FIRST value of the quasimomentum
		u: 	(n,) numpy array, used to normalise the output eigenvector in the NLII scheme
		wInit: 	(complex) float, starting guess for the eigenvalue that the solver will use for the FIRST value of the quasimomentum.
		t1End: 	(optional) float - default np.pi, the range for theta1, the first component of the quasimomentum, is taken to run over the range [-t1End, t1End]
		t2End: 	(optional) float - default np.pi, the range for theta2, the second component of the quasimomentum, is taken to run over the range [-t2End, t2End]
		tPts: 	(optional) int - default 1000, the intervals for theta1 and theta2 will be divided into tPts+1 mesh points, with the generalised eigenvalue problem being solved at each. NOTE: must be even!
		fileName: 	(optional) str - default False, if a string is supplied then an output CSV file will be written under the supplied filename. .csv extension will be appended if not provided.
	**kwargs:
		These will be passed into NLII
	OUTPUTS:
		storeDict: 	dictionary, with keys correspond to the run number of each entry, with the value being a list of the following;
			- quasimomentum value during solve - (2,) numpy array, 
			- starting data run number - int, 
			- found e'val - float, 
			- found e'vector components - (n,) numpy array depending on M-matrix size,
			- list of flagged convergence issues
			The initial solve has 0 in the place of the "run number of starting guess"
		fileName.csv: 	(optional) .csv file - default not produced. If a fileName is provided, the .csv file is structured so that each row number corresponds to the run number, with the first row being run 0 (initial solve). The remainder of the row is as follows;
			- qm1, 
			- qm2, 
			- run number of initial guess, 
			- e'val found, 
			- e'vec components,
			- bool 0/1, if 1 then there were convergence issues flagged during solve
			The initial solve has 0 in the place of the "run number of starting guess"
	'''
	
	#the plan is to output a file or dictionary with each row/key as an integer. We will call this integer the "run number", and it will just be the order in which solves were undertaken, with run number 0 being the initial solve.
	#after the initial solve, we will use the solution from the previous solve as initial data for the next solve, and will record the run number from which we got the starting guess for each solve:
	#For a dictionary output, each record is as follows:
	## key = run number, with the value being a list of the following;
	## quasimomentum value during solve - (2,) numpy array, starting data run number - int, found e'val - float, found e'vector components (n,) numpy array depending on M-matrix size
	## the initial solve has 0 in the place of "run number of starting guess"
	storeDict = {}
	#For a file output, we write to csv
	## row - the row corresponds to the run number, with the first row being run 0 (initial solve)
	## run number of initial guess, qm1, qm2, e'val found, e'vec components
	if isinstance(fileName, str):
		#we want to output to file too
		fileOut = True
		if fileName[-4:]!='.csv':
			fName = fileName + '.csv'
		else:
			fName = fileName
	else:
		fileOut = False
	
	#for each thetaVal, these functions return another function that is the M-matrix (respectively it's derivative) evaluated at w, thetaVal.
	#Explictly, f(thetaVal)(w) = M(w,thetaVal), df(thetaVal)(w) = dM(w,thetaVal). Note that f(thetaVal) is a function, as is df(thetaVal).
	f = lambda thetaVal: CreateHandle(M, thetaVal)
	df = lambda thetaVal: CreateHandle(dM, thetaVal)

	if tPts%2!=0:
		warn('tPts is not an even number, aborting solve.')
		return
	else:
		t1Pts = tPts;	t2Pts = tPts;
		half1 = int(t1Pts/2); half2 = int(t2Pts/2);
	
	t1Vals = np.linspace(-t1End,t1End,num=t1Pts+1,endpoint=True) 
	t2Vals = np.linspace(-t2End,t2End,num=t2Pts+1,endpoint=True)	

	runNumber = 0
	#initial solve
	t0 = np.asarray([t1Vals[half1], t2Vals[half2]])
	wFirst, vFirst, conIss, nItts = NLII(f(t0), df(t0), vInit, u, wInit, conLog=True, retNItt=True, **kwargs)

	#check for a failed solve because that would be bad
	if conIss: #empty list is implicitly false in python
		#warnings in first solve
		warn('Warnings given in first solve, consider alternative start point/ abortion of testing')
	#assign initial data to dictionary
	storeDict[runNumber] = [t0, 0, wFirst, vFirst, conIss]
	if fileOut:
		#save to file if we want this
		WriteRow(fName, t0, 0, wFirst, vFirst, conIss)
			
	#if we get to here, the first solve went OK so now we need to step through the thetas and do the solving thing. Theta is 2D to make things more complicated though, so we need to be clever about how we step through the values to make sure we are using a starting value that comes from a "close" theta.
	print('NOTE: Current program does not check for failed convergence after first solve - take care when using outputs')
	#the easiest way I can see to do this is to use a "spiral" technique to map out all the theta values, our initial solve takes care of the "central" solve value. We can now envisage theta as occupying a finite square of Z^2, with our run number 0 solve at (0,0). Thus, we can now "spiral" out from (0,0), using the previous run as the approximation to our next run.
	#we have half1 points in the "x1" axis, and half2 points in the "x2" axis each side of zero, so our spiral will have to terminate at the lower of these two values. After which, we will have to be clever about how we complete the solve.
	sqVal = min(half1, half2)
	for i in range(1, (2*sqVal+1)**2):
		#square i uses indices 
		tI = Spiral(i)
		qm = np.asarray([t1Vals[half1 + tI[0]], t2Vals[half2 + tI[1]]])
		print('------------- \nNow solving for QM: (%.5f, %.5f)' %(qm[0], qm[1]))
		wStart = storeDict[i-1][2] 
		#there may be an issue here - if f(qm) is much greater in norm than df(qm), then the first NLII step will adjust the eigenvalue by a large amount (as u*v_k/u*x will be large as x is small compared to v_k in this case), and so we will obtain an eigenvalue far away from our previous eigenvalue estimate. This occurs when we are near a stationary point of M's entries, for example.
		#to counter this, we can check the relative sizes of f and df before calling NLII at the starting values from the previous run. If this problem occurs, then we can instead start from an alternative start point?
		if norm(f(qm)(wStart))/norm(df(qm)(wStart)) >= 1e4:
			#this is the bad case, so use the initial data for the solve
			print('Matrix norm mismatch')
			wFound, vFound, conIss = NLII(f(qm), df(qm), vInit, u, wInit, conLog=False, retNItt=False, **kwargs)[0:3]
			storeDict[i] = [qm, 0, wFound, vFound, conIss]
			if fileOut:
				WriteRow(fName, qm, 0, wFound, vFound, conIss)
		elif norm(f(qm)(wStart)) <= 1e-8:
			#starting value for M-matrix is almost zero, which will be flagged as singular, use a different start point
			print('Matrix zero at start value')
			wFound, vFound, conIss = NLII(f(qm), df(qm), vInit, u, wInit, conLog=False, retNItt=False, **kwargs)[0:3]
			storeDict[i] = [qm, 0, wFound, vFound, conIss]
			if fileOut:
				WriteRow(fName, qm, 0, wFound, vFound, conIss)
		else:
			#previous run is used as starting guess
			vStart = storeDict[i-1][3]
			wFound, vFound, conIss = NLII(f(qm), df(qm), vStart, u, wStart, conLog=False, retNItt=False, **kwargs)[0:3]
			storeDict[i] = [qm, i-1, wFound, vFound, conIss]
			if fileOut:
				WriteRow(fName, qm, i-1, wFound, vFound, conIss)
		print('Found eigenvalue: %.5e + %.5e' %(np.real(wFound), np.imag(wFound)))
	return storeDict

def TestMatrix(w, theta):
	'''
	A test case, to determine whether the script is actually performing what I expect it to
	'''
	
	M = np.diag([theta[0]-w, theta[1]-w, theta[0]+theta[1]-w])
	
	return M

def TestMatrixDeriv(w, theta):
	'''
	A test case, to determine whether the script is actually performing what I expect it to
	'''
	
	dM = np.diag([-1, -1, -1])
	
	return dM

def PlottingStuff(storeDict):
	
	k = list(storeDict.keys())
	xAxis = []
	yAxis = []
	for key in storeDict.keys():
		if k[0][0]==key[0]:
			xAxis.append(key[1]) #theta2 value
			yAxis.append(storeDict[key][0]) #eigenvalue converged to
	
	xAxis = np.asarray(xAxis)
	yAxis = np.asarray(yAxis)
	
	plt.plot(xAxis, yAxis)
	plt.show()
	
	return

def SurfFromFile(fileName, plotName=None, **kwargs):
	'''
	Read the (QM, omega) data from the given file and produce a surface plot of the eigenvalues as a function of the quasi-momentum.
	INPUTS:
		fileName: 	path to file, the .csv file which the (QM, omega) data is saved, in the format of the output of SolveStepSolve
		plotName: 	(optional) str - default None, if provided the created surface plot will be saved with this name (extension needs to be provided), otherwise the plot will be saved as a pdf under the same name as fileName.
	OUTPUTS:
		fig, ax: 	Axes3D objects, handles for the 3D figure that was produced by the program
	**kwargs:
		Arguments to be passed to the internal call to Axes3D.plot_trisurf
	'''
	
	if isinstance(plotName, str):
		saveStr = plotName
	else:
		saveStr = fileName[:-4] + '.pdf'
	
	qm1, qm2, omegaVals = np.loadtxt(fileName, dtype='complex, complex, complex', delimiter=',', usecols=(0,1,3), unpack=True)
	
	#we should only have real QM values, and real omega values, so we now just check that we indeed have this before niavely making this assumption and beginning to plot
	if norm(np.imag(qm1))>1e-8:
		#bad values - qm1 has imaginary components for some reason!
		raise ValueError('qm1 has imaginary entries - review file input.')
	else:
		#safe to discard imaginary parts and cast to real
		qm1 = np.real(qm1)
	if norm(np.imag(qm2))>1e-8:
		#bad values - qm1 has imaginary components for some reason!
		raise ValueError('qm2 has imaginary entries - review file input.')
	else:
		#safe to discard imaginary parts and cast to real
		qm2 = np.real(qm2)
	if norm(np.imag(omegaVals))>1e-8:
		#bad values - qm1 has imaginary components for some reason!
		raise ValueError('omegaVals has imaginary entries - review file input.')
	else:
		#safe to discard imaginary parts and cast to real
		omegaVals = np.real(omegaVals)
		
	#disable interactive plotting so the figure is not rendered in the console
	plt.ioff()
	#now we can create the surface plot
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot_trisurf(qm1[omegaVals<=4], qm2[omegaVals<=4], omegaVals[omegaVals<=4], linewidth=0.2, antialiased=True, **kwargs)
	#ax.plot_trisurf(qm1, qm2, omegaVals, linewidth=0.2, antialiased=True, **kwargs)
	ax.set_xlabel(r'$\theta_1$')
	ax.set_ylabel(r'$\theta_2$')
	ax.set_zlabel(r'$\omega$')
	plt.savefig(saveStr)
	plt.close(fig)
	#enable interactive plotting again
	plt.ion()
	
	return fig, ax

def DataFromFile(fileName):
	'''
	Read the (QM, omega) data from the given file and return arrays containing the data.
	INPUTS:
		fileName: 	path to file, the .csv file which the (QM, omega) data is saved, in the format of the output of SolveStepSolve
	OUTPUTS:
		qm: 	(2,n) float numpy arrays, each column is a value of the quasi-momentum parameter used in a the run indexed at runNo[j]
		runNos: 	(n,) int numpy arrays, run number of each solve
		eVals: 	(2,n) float numpy array, the eigenvalue found during runNo[j]. Will error if complex eigenvalues were found
		eVecs: 	(3,n) complex numpy array, the eigenvectors found during runNo[j] - stored column-wise
		conIss: (n,) bool numpy array, if runNo[j] flaged convergence issues 
	'''

	runNos, eVals, conIss = np.loadtxt(fileName, dtype='complex, complex, complex', delimiter=',', usecols=(2,3,7), unpack=True)
	#check imports have sensible values
	if norm(np.imag(runNos))>1e-8:
		#run numbers are complex - something is bad!
		raise ValueError('Run numbers are complex')
	else:
		runNos = np.real(runNos)
		runNos = runNos.astype(int) #should be safe to convert
		nRuns = len(runNos)
	if norm(np.imag(conIss))>1e-8:
		#conIss is complex - something is very bad!
		raise ValueError('Convergence issues flagged as imaginary values')
	else:
		conIss = np.real(conIss).astype(bool)
	if norm(np.imag(eVals))>1e-8:
		#bad values - qm1 has imaginary components for some reason!
		raise ValueError('eVals has imaginary entries - review file input.')
	else:
		#safe to discard imaginary parts and cast to real
		eVals = np.real(eVals)
	
	#preallocate multi-dimensional arrays
	qm = np.zeros((2,nRuns), dtype=complex)
	eVecs = np.zeros((3,nRuns), dtype=complex)
	
	#import qms and eigenvectors
	qm[0,:], qm[1,:], eVecs[0,:], eVecs[1,:], eVecs[2,:] = np.loadtxt(fileName, dtype='complex, complex, complex, complex, complex', delimiter=',', usecols=(0,1,4,5,6), unpack=True)
	
	#we should only have real QM values
	if not ( norm(np.imag(qm[0,:]))>1e-8 and norm(np.imag(qm[1,:]))>1e-8 ):
		#all is good, can discard imaginary parts and use a float array instead
		qm = np.real(qm)
	elif norm(np.imag(qm[0,:]))>1e-8:		
		#bad values - qm[0,:] has imaginary components for some reason!
		raise ValueError('qm[0,:] has imaginary entries - review file input.')
	elif norm(np.imag(qm[1,:]))>1e-8:
		#bad values - qm[1,:] has imaginary components for some reason!
		raise ValueError('qm[1,:] has imaginary entries - review file input.')
	else:
		#just in case we somehow get here without triggering the other catches
		raise ValueError('qm has imaginary values, but could not pinpoint bad component')

	
	return qm, runNos, eVals, eVecs, conIss

if __name__=='__main__':
	
	G_TFR = TFR_Setup()
	M_TFR, dM_TFR, invDenom = G_TFR.ConstructAltM()
	
	M_Test = TestMatrix
	dM_Test = TestMatrixDeriv
	
	v0 = np.asarray([1.,1.,1.]) / np.sqrt(3)
	u = UnitVector(2)
	w0 = np.pi/2