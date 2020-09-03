#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:08:29 2020

@author: will

Various test functions for the Quantum graph solvers that I have written. Includes:
	M_TFR_Exact
"""

import numpy as np
from numpy import sin, cos

from AuxMethods import cot, csc, AbsLess

from VertexClass import Vertex
from EdgeClass import Edge
from GraphClass import Graph

def M_TFR_Exact(w, theta=np.zeros((2,))):
	'''
	Exact M-matrix for the TFR problem
	INPUTS:
		w 	: float, value of w
		theta 	: (optional) (2,) numpy array, value of the quasimomentum parameters
	OUTPUTS:
		mat 	: M-matrix for the TFR problem, evaluated at (w,theta)
	'''
	
	mat = np.zeros((3,3), dtype=complex)
	mat[0,0] = 2*w*cot(w/2)
	mat[1,1] = 2*w*cot(w/2)
	mat[2,2] = 4*w*cot(w/2)
	mat[0,2] = -2*w*np.cos(theta[1]/2)*csc(w/2)
	mat[2,0] = -2*w*np.cos(theta[1]/2)*csc(w/2)
	mat[1,2] = -2*w*np.cos(theta[0]/2)*csc(w/2)
	mat[2,1] = -2*w*np.cos(theta[0]/2)*csc(w/2)
	
	return mat

def AltM_TFR_Exact(w, theta=np.zeros((2,))):
	'''
	Exact AltM-matrix for the TFR problem. We have M = w/sin(w/2) * AltM.
	INPUTS:
		w 	: float, value of w
		theta 	: (optional) (2,) numpy array, value of the quasimomentum parameters
	OUTPUTS:
		mat 	: M-matrix for the TFR problem, evaluated at (w,theta)
	'''	

	mat = np.zeros((3,3), dtype=complex)
	mat[0,0] = 2*cos(w/2)
	mat[1,1] = 2*cos(w/2)
	mat[2,2] = 4*cos(w/2)
	mat[0,2] = -2*cos(theta[1]/2)
	mat[2,0] = -2*cos(theta[1]/2)
	mat[1,2] = -2*cos(theta[0]/2)
	mat[2,1] = -2*cos(theta[0]/2)
	
	return mat

def dAltM_TFR_Exact(w, theta=np.zeros((2,))):
	'''
	Exact d/dw (AltM ) for the TFR problem.
	INPUTS:
		w 	: float, value of w
		theta 	: (optional) (2,) numpy array, value of the quasimomentum parameters
	OUTPUTS:
		mat 	: dM-matrix for the TFR problem, evaluated at (w,theta)
	'''	

	mat = np.zeros((3,3), dtype=complex)
	mat[0,0] = -sin(w/2)
	mat[1,1] = -sin(w/2)
	mat[2,2] = -2*sin(w/2)

	return mat

def TFR_Setup(alpha=0.):
	'''
	Returns a Graph instance with the TFR geometry
	INPUTS:
		alpha: 	(optional) float, value of the coupling constant to place at the central vertex. Default 0.0
	OUTPUTS:
		G: 	Graph instance, setup as the TFR (cross) problem with coupling constant alpha at the central vertex
	'''
	
	v1 = Vertex(1, np.asarray([0,0],dtype=float))
	v2 = Vertex(2, np.asarray([1,0],dtype=float))
	v3 = Vertex(3, np.asarray([0.5,0],dtype=float), coupConst=alpha)
	
	e13 = Edge(v1, v3, length=0.5, qm=np.asarray([0,-1],dtype=float))
	e31 = Edge(v3, v1, length=0.5, qm=np.asarray([0,-1],dtype=float))
	e23 = Edge(v2, v3, length=0.5, qm=np.asarray([-1,0],dtype=float))
	e32 = Edge(v3, v2, length=0.5, qm=np.asarray([-1,0],dtype=float))
	
	V = [v1, v2, v3]
	E = [e13, e31, e23, e32]
	
	G = Graph(V,E)
	
	return G

def CompareOverRange(f1, f2, xRange='auto', tol=1e-8):
	'''
	Given two (posibly matrix-valued) functions f1, f2 of an argument x, compare the function outputs at a range of points over an interval and determine places where they disagree by a given tolerance.
	Primary use of this function is to check that computed M-matrices match up to those we know analytically.
	INPUTS:
		f1, f2: 	lambda functions, each taking a single argument x and returning numpy arrays of identical shape to each other
		xRange: 	(optional) numpy array, range of points x at which to test agreement between f1 and f2. Default [-pi,pi] with 1000 linearly spaced points
		tol: 	(optional) float, tolerance for agreement. Default 1e-8
	OUTPUTS:
		nFail: 	int, number of points x at which significant disagreement was found between f1 and f2
		xLog: 	numpy array, array of points x at which significant disagreement was found between f1 and f2
	'''
	
	if isinstance(xRange, str):
		#xRange not supplied, so use the default value
		xR = np.linspace(-np.pi, np.pi, 1000)
	else:
		xR = xRange
		
	xBad = [];	nFail = 0
	
	for x in xR:
		f1Val = f1(x);	f2Val = f2(x)
		#check if outputs are same shape first!
		if np.shape(f1Val) != np.shape(f2Val):
			print('x:', x)
			raise ValueError('Functions return different sized outputs')
		else:
			#if they are the same shape, work out how many elements there are in each that need to match
			nDims = np.shape(np.shape(f1Val))[0]
			nElements = 1
			for dim in range(nDims):
				nElements *= np.shape(f1Val)[dim]
			#now check agreement between all the elements
			nElementsAgree = np.sum( AbsLess(f1Val - f2Val, tol) )
			if nElementsAgree < nElements:
				#not all elements of the outputs agree, flag this point for further investigation
				xBad.append(x)
				nFail += 1
	#now convert xBad into xLog and return
	xLog = np.asarray(xBad)
	
	return nFail, xLog

def CompareOverTheta(f1, f2, tPts=1000):
	'''

	'''

	t1Range = np.linspace(-np.pi, np.pi, num=tPts)
	t2Range = np.linspace(-np.pi, np.pi, num=tPts)
	issueDict = {}
	for i in range(tPts):
		print('%d, ' % (i), end='')
		if i%25==0:
			print('\n', end='')
		for j in range(tPts):
			qm = np.asarray([t1Range[i], t2Range[j]])
			g1 = lambda x: f1(x, theta=qm)
			g2 = lambda x: f2(x, theta=qm)
			nFail, wLog = CompareOverRange(g1, g2)
			if nFail!=0:
				issueDict[i,j] = [qm, nFail, wLog]
	return issueDict