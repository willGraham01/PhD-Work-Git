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
	- NLII (to be added)
	
"""

import numpy as np
from numpy import sin, tan

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
