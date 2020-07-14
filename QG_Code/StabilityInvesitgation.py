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

from warnings import warn

from VertexClass import Vertex
from EdgeClass import Edge
from GraphClass import Graph

from AuxMethods import NLII, EvalCheck, UnitVector
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

def SolveStepSolve(M, dM, vInit, u, wInit, t1End = np.pi, t2End = np.pi, tPts = 1000):
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
	OUTPUTS:
		storeDict: 	dictionary, keys are the quasimomentum pairs for which eigenpairs (w,v) were found (or attempted to be found). Each value is a list [w, v, nItts, w0, v0] consisting of (at the value of theta as in the corresponding key)
			w - the eigenvalue that was found 
			v - the eigenvector that was found
			nItts - the number of NLII iterations that were performed
			w0 - initial guess for the eigenvalue that was used
			v0 - initial guess for the eigenvector that was used
	'''
	
	#for the time being, store the relevant data in a dictionary. The keys of the dictionary will be the values of theta that were used, and each storeDict[t1,t2] entry contains a list consisting of [w, v, nItts, w0, v0] in that order. The initial entry also has an additional list element specifying that it was this
	storeDict = {}
	
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

	#initial solve
	t0 = np.asarray([t1Vals[half1], t2Vals[half2]])
	wFirst, vFirst, conIss, nItts = NLII(f(t0), df(t0), vInit, u, wInit, conLog=True, retNItt=True)
	storeDict[t1Vals[half1], t2Vals[half2]] = [wFirst, vFirst, nItts, wInit, vInit, 'StartingThetaValue']
	
	#check for a failed solve because that would be bad
	if not conIss: #empty list is implicitly false in python
		#error in first solve
		warn('Error in first solve, abort testing')
		return storeDict
	
	#if we get to here, the first solve went OK so now we need to step through the thetas and do the solving thing. Theta is 2D to make things more complicated though, so we need to be clever about how we step through the values to make sure we are using a starting value that comes from a "close" theta.
	print('NOTE: Current program does not check for failed convergence after first solve - take care when using outputs')
	#first, let's fill out the whole range of theta1 values for when theta2 is 0, the initial theta2 value
	for i1 in range(half1):
		#first step "right"
		#get start point from nearest solved system
		w0 = storeDict[t1Vals[half1 + i1], t2Vals[half2]][0]
		v0 = storeDict[t1Vals[half1 + i1], t2Vals[half2]][1]
		theta = np.asarray([t1Vals[half1 + (i1+1)], t2Vals[half2]])
		wFound, vFound, conIss, nItts = NLII(f(theta), df(theta), v0, u, w0, conLog=False, retNItt=True)
		storeDict[t1Vals[half1 + (i1+1)], t2Vals[half2]] = [wFound, vFound, nItts, w0, v0]
		#now step "left"
		#get start point from nearest solved system
		w0 = storeDict[t1Vals[half1 - i1], t2Vals[half2]][0]
		v0 = storeDict[t1Vals[half1 - i1], t2Vals[half2]][1]
		theta = np.asarray([t1Vals[half1 - (i1+1)], t2Vals[half2]])
		wFound, vFound, conIss, nItts = NLII(f(theta), df(theta), v0, u, w0, conLog=False, retNItt=True)
		storeDict[t1Vals[half1 - (i1+1)], t2Vals[half2]] = [wFound, vFound, nItts, w0, v0]	
	#storeDict now contains the eigenpair that was found for each [t1,0] theta pair. Let's do the same for each [0,t2] pair
	for i2 in range(half2):
		#first step "up"
		w0 = storeDict[t1Vals[half1], t2Vals[half2 + i2]][0]
		v0 = storeDict[t1Vals[half1], t2Vals[half2 + i2]][1]
		theta = np.asarray([t1Vals[half1], t2Vals[half2 + (i2+1)]])
		wFound, vFound, conIss, nItts = NLII(f(theta), df(theta), v0, u, w0, conLog=False, retNItt=True)
		storeDict[t1Vals[half1], t2Vals[half2 + (i2+1)]] = [wFound, vFound, nItts, w0, v0]
		#now step "down"
		w0 = storeDict[t1Vals[half1], t2Vals[half2 - i2]][0]
		v0 = storeDict[t1Vals[half1], t2Vals[half2 - i2]][1]
		theta = np.asarray([t1Vals[half1], t2Vals[half2 - (i2+1)]])
		wFound, vFound, conIss, nItts = NLII(f(theta), df(theta), v0, u, w0, conLog=False, retNItt=True)
		storeDict[t1Vals[half1], t2Vals[half2 - (i2+1)]] = [wFound, vFound, nItts, w0, v0]
	#storeDict now contains solutions for every pair [t1,0] and [0,t2]. Now we can loop over values...
	
	for i1 in range(half1):
		for i2 in range(half2):
			#step right, then up - we are solving for theta = [t1[t1Pts/2 + (i1+1)], t2[t2Pts/2 + (i2+1)]] using the answer to theta = [t1[t1Pts + i1], t2[t1Pts + i2]]
			#get start point from nearest solved system
			w0 = storeDict[t1Vals[half1 + i1], t2Vals[half2 + i2]][0]
			v0 = storeDict[t1Vals[half1 + i1], t2Vals[half2 + i2]][1]
			theta = np.asarray([t1Vals[half1 + (i1+1)], t2Vals[half2 + (i2+1)]])
			wFound, vFound, conIss, nItts = NLII(f(theta), df(theta), v0, u, w0, conLog=False, retNItt=True)
			storeDict[t1Vals[half1 + (i1+1)], t2Vals[half2 + (i2+1)]] = [wFound, vFound, nItts, w0, v0]
			#step right, then down - we are solving for theta = [t1[t1Pts/2 + (i1+1)], t2[t2Pts/2 - (i2+1)]] using the answer to theta = [t1[t1Pts + i1], t2[t1Pts - i2]]
			w0 = storeDict[t1Vals[half1 + i1], t2Vals[half2 - i2]][0]
			v0 = storeDict[t1Vals[half1 + i1], t2Vals[half2 - i2]][1]
			theta = np.asarray([t1Vals[half1 + (i1+1)], t2Vals[half2 - (i2+1)]])
			wFound, vFound, conIss, nItts = NLII(f(theta), df(theta), v0, u, w0, conLog=False, retNItt=True)
			storeDict[t1Vals[half1 + (i1+1)], t2Vals[half2 - (i2+1)]] = [wFound, vFound, nItts, w0, v0]
			#step left, then down - we are solving for theta = [t1[t1Pts/2 - (i1+1)], t2[t2Pts/2 - (i2+1)]] using the answer to theta = [t1[t1Pts - i1], t2[t1Pts - i2]]
			w0 = storeDict[t1Vals[half1 - i1], t2Vals[half2 - i2]][0]
			v0 = storeDict[t1Vals[half1 - i1], t2Vals[half2 - i2]][1]
			theta = np.asarray([t1Vals[half1 - (i1+1)], t2Vals[half2 - (i2+1)]])
			wFound, vFound, conIss, nItts = NLII(f(theta), df(theta), v0, u, w0, conLog=False, retNItt=True)
			storeDict[t1Vals[half1 - (i1+1)], t2Vals[half2 - (i2+1)]] = [wFound, vFound, nItts, w0, v0]
			#step left, then up - we are solving for theta = [t1[t1Pts/2 - (i1+1)], t2[t2Pts/2 + (i2+1)]] using the answer to theta = [t1[t1Pts - i1], t2[t1Pts + i2]]
			w0 = storeDict[t1Vals[half1 - i1], t2Vals[half2 + i2]][0]
			v0 = storeDict[t1Vals[half1 - i1], t2Vals[half2 + i2]][1]
			theta = np.asarray([t1Vals[half1 - (i1+1)], t2Vals[half2 + (i2+1)]])
			wFound, vFound, conIss, nItts = NLII(f(theta), df(theta), v0, u, w0, conLog=False, retNItt=True)
			storeDict[t1Vals[half1 - (i1+1)], t2Vals[half2 + (i2+1)]] = [wFound, vFound, nItts, w0, v0]
	#this should complete storeDict - we can now return what we have created...
	
	return storeDict

if __name__=='__main__':
	
	G_TFR = TFR_Setup()
	M_TFR, dM_TFR = G_TFR.ConstructM(derivToo=True)