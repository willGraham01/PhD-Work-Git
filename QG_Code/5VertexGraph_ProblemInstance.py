#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 18:00:14 2020

@author: will

File contains a class that defines a ProblemInstance.
Each ProblemInstance takes in a value for the quasi-momentum and the length parameters l1,...,l4, and then can construct a function detM which evaluates the expression for the determinant of the M-matrix, for that problem at the specified value of the efFreq.
"""

import sys
sys.path.append('../')

import numpy as np
from numpy import sqrt, sin, cos

import matplotlib.pyplot as plt

#from AuxPlotFns import FastFunctionPlot as FFP
from AuxMethods import csc, cot, AbsLess

#edge lengths
#l1 = 0.25;	l2 = 0.25;	l3 = 0.25;	l4 = 0.25; lengths=np.asarray([l1,l2,l3,l4])
l1 = 0.5;	l2 = 0.5;	l3 = 0.;	l4 = 0.; lengths=np.asarray([l1,l2,l3,l4])
qm = np.asarray([0,0],dtype=float)

class ProblemInstance:
	'''
	One instance of the 5-vertex-general-graph problem, at a fixed value of the quasi-momentum.
	INITIASLISE WITH:
		p = ProblemInstance(lengths, qm)
	PROPERTIES:
		lVals: 	(4,) float numpy array, the values of the parameters l1,...,l4
		gLens: 	(4,) float numpy array, the values of the graph edges l12, l23, l24, l31 in that order
		qm: 	(2,) float numpy array, the value of the quasi-momentum used in this instance of the problem
		R12, R23, R24, R31: 	(2,2) float numpy array, the rotation matrices of the graph
		edgeQM: 	(4,) float numpy array, the value of (R_{jk}qm_{jk})_2 for each edge
	METHODS:
	
	'''
	
	def __init__(self, lengths, qm):
		'''
		Initialiser method for ProblemInstance
		INPUTS:
			lengths: 	(4,) float numpy array, contains the values of the parameters l1,l2,l3,l4 to be used
			qm: 	(2,) float numpy array, value of the quasi-momentum to use for this instance of the problem
		OUTPUTS:
			Instance of ProblemInstance with the attributes generated from the input values
		'''
		#checks for valid inputs
		if (lengths[0]+lengths[3]>1):
			raise ValueError('l1+l4>1, aborting')
		elif (lengths[1]+lengths[2]>1):
			raise ValueError('l2+l3>1, aborting')
			
		#assign edge lengths
		self.lVals = lengths
		l1 = lengths[0];	l2 = lengths[1];	l3 = lengths[2];	l4 = lengths[3]
		#quasi-momentm
		self.qm = qm
		#graph edge lengths
		l12 = sqrt(l2*l2 + l4*l4);	l23 = sqrt(l3*l3 + (l1+l4)*(l1+l4))
		l24 = sqrt(l3*l3 + (1-l1-l4)*(1-l1-l4)); l31 = sqrt(l1*l1 + (1-l2-l3)*(1-l2-l3))
		self.gLens = np.asarray([l12, l23, l24, l31], dtype=float)
		#rotation matrices
		self.R12 = np.asarray([[l4, -l2], [l2, l4]], dtype=float) / l12
		self.R23 = np.asarray([[-l1-l4, l3],[-l3, -l1-l4]], dtype=float) / l23
		self.R24 = np.asarray([[1-l1-l4, -l3],[l3, 1-l1-l4]], dtype=float) / l24
		self.R31 = np.asarray([[l1, l2+l3-1],[1-l2-l3, l1]], dtype=float) / l31
		#quasi-momentum on the edges
		qm12 = np.matmul(self.R12, qm)[1]
		qm23 = np.matmul(self.R23, qm)[1]
		qm24 = np.matmul(self.R24, qm)[1]
		qm31 = np.matmul(self.R31, qm)[1]
		self.edgeQM = np.asarray([qm12, qm23, qm24, qm31], dtype=float)
		
		return #end init method
		
	def MatrixDet(self):
		'''
		Creates a function that can evaluate the determinant of the M-matrix as a function of the effective frequency
		INPUTS:
			qm: 	Quasi-momentum as a (2,) numpy array.
		OUTPUTS:
			DetM: 	Function, evaluates the determinant of the M-matrix as a function of effFreq.
		'''

		#beta values for ease of computation
		bVals = self.gLens * self.edgeQM
		gL = self.gLens
	
		def DetM(effFreq):
			'''
			INPUTS:
				effFreq: 	Effective frequency sqrt(omega^2-wavenumber^2). Note: omega>wavenumber
			OUTPUTS:
				matDet: 	Value of the M-matrix determinant at the given point.
			'''
			matDet = 0
			cotVals = cot(effFreq * gL)
			cscVals = csc(effFreq * gL)
			cosVals = cos(effFreq * gL)
			
			matDet += -( 2 + np.sum(cotVals) ) #-2-sum of cot values
			matDet += 2 * np.prod(cscVals.take([0,1,3])) * \
			( np.prod( cosVals.take([0,1,3]) ) - cos( np.sum( bVals.take([0,1,3]) ) ) ) #2csc12csc23csc31*(cos12cos23cos31 - cos(b12+b23+b31))
			matDet += 2 * np.prod(cscVals.take([0,2,3])) * \
			( np.prod( cosVals.take([0,2,3]) ) - cos( np.sum( bVals.take([0,2,3]) ) ) ) #2csc12csc24csc31*(cos12cos24cos31 - cos(b12+b24+b31))
			matDet += 2 * np.prod(cscVals.take([1,2])) * ( np.prod( cosVals.take([1,2]) ) - cos(bVals[1]-bVals[2]) ) #2csc23csc24*(cos23cos24 - cos(b23-b24))
			return matDet
		
		return DetM

#end of class definition

def WorkedEquation(effFreq):
	'''
	For when l1=l2=0.5, l3=l4=0, this is the equivalent expression for effFreq to give an e'value when it's between the values -3 and 1.5.
	INPUTS:
		effFreq: 	(n,) float numpy array, values of effFreq to use
	OUTPUTS:
		val: 	(n,) float numpy array, values of the equation
	'''
	val = 0;	r2 = sqrt(2); rm = (r2-1)/r2;	rp = (r2+1)/r2
	val += (1/2)*cos(effFreq/r2)
	val += (3/8)*cos(rm*effFreq)
	val += (9/8)*cos(rp*effFreq)
	val += (1/2)*sin(rp*effFreq)
	val += -(1/2)*sin(rm*effFreq)
	
	return val

if __name__=='__main__':
	
	wPts = 25000
	w = np.linspace(0,15*np.pi,wPts)
	vals = WorkedEquation(w)
	eValInds = AbsLess(vals, -3, 1.5)
	
	fig, ax = plt.subplots(1)
	ax.plot(w/np.pi, vals)
	ax.plot(w/np.pi, -3*np.ones((wPts,)), 'k')
	ax.plot(w/np.pi, 1.5*np.ones((wPts,)), 'k')
	ax.scatter(w[eValInds]/np.pi, np.zeros_like(w[eValInds], dtype=float), s=1, c='r', marker='x')
	ax.set_xlabel(r'$\frac{\Lambda}{\pi}$')
	ax.set_ylabel(r'Dispersion Expression Value')
	ax.set_xlim([w[0]/np.pi, w[-1]/np.pi])
	saveStr = input('Save figure as (.pdf appended automatically): ') + '.pdf'
	if len(saveStr)<=4:
		print('File not saved (invalid filename)')
	else:
		print('Filename provided: ', saveStr)
		fig.savefig(saveStr)