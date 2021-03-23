#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 08:31:12 2020

@author: will

Script and functions to numerically compute the (Integrated) Density of States for the TFR thick-vertex cross-in-plane setup, see section 2.2 of the TFR.

Equation (2.7) in section 2.2 gives that eigenvalues w are those which solve the following equation:
	-(alpha*w/4)sin(w) + cos(w) = cos((theta1+theta2)/2)cos((theta1-theta2)/2)
This will necesitate the use of fsolve to determine the eigenvalues which a given theta (quasimomentum) corresponds to.
Propositions 2.2 and 2.3 in section 2.2 also demonstrate that each "band" I_n is contained in the interval [(n-1)pi, n*pi], so we can utilise this knowledge to compute the Integrated DoS in over each band sequentally. Furthermore, we know that the right-endpoint of each band is precisely n*pi.

A recommended call to Visual_FullDoS is along the lines of
f = Visual_FullDoS(alpha, lastBand=n, nPts=100, dx=1e-1)
where n is the index of the final spectral band that you want to be included in the plot. 
This call is relatively fast (due to low nPts) and also produces relatively good quality for bands up to n=5.
More detail on the choice of parameters can be found in the function comments themselves.

All functions have a docstring that can be printed by calling help() on that function, EG help(LHS).
Test and visualisation functions begin with either Test_ or Visual_.
Contructor functions begin with caps and are CamelCased.
Variables begin with lowercase letters and are camelCased, with exceptions for variables that are acronyms
"""

from warnings import warn

import numpy as np
from numpy import sin, cos
from numpy import pi
from numpy import linspace

from scipy.optimize import fsolve

import matplotlib.pyplot as plt

def LHS(w, alpha):
	'''
	Left-hand side of equation (2.7), evaluated at a point w for a given alpha.
	Inputs:
		w: 	(n,) float numpy array, set of real frequencies or eigenvalues to evaluate RHS at
		alpha: 	float, value of the vertex-scaling parameter alpha
	Outputs:
		LHSVals: (n,) float numpy array, evaluation of the LHS
	'''
	
	LHSVals = -(alpha/4) * w * sin(w) + cos(w) #default np array ops are element-wise
	return LHSVals

def LHSDeriv(w, alpha):
	'''
	Derivate (wrt w) of the left-hand side of equation (2.7), evaluated at a point w for a given alpha.
	Inputs:
		w: 	(n,) float numpy array, set of real frequencies or eigenvalues to evaluate RHS at
		alpha: 	float, value of the vertex-scaling parameter alpha
	Outputs:
		LHSDerivVals: (n,) float numpy array, evaluation of the derivative of the LHS
	'''
	
	LHSVals = ((alpha/4)*w)*cos(w) + (alpha/4 - 1)*sin(w) #default np array ops are element-wise
	return LHSVals

def ProductCos(theta):
	'''
	Right-hand side of equation (2.7), evaluated at a given quasimomentum theta in the range (-pi,pi]^2
	Inputs:
		theta: 	(2,) float numpy array, value of the quasimomentum to evaluate the LHS at
	Outputs:
		RHSVals: (n,) float numpy array, evaluation of the RHS product-cos term
	'''	

	thetaSum = theta.sum(axis=0) / 2. #(theta1+theta2)/2, for each column
	thetaDiff = (theta[0] - theta[1]) / 2. #(theta1-theta2)/2, for each column
	RHSVals = cos(thetaSum) * cos(thetaDiff)
	return RHSVals

def EstimateBandStart(bandIndex, alpha, nPts=1000, precise=False):
	'''
	Given an integer bandIndex, estimate the beginning of the band I_bandIndex by testing where the LHS of (2.7) has absolute value no greater than 1. Return the estimated start point of the band, which is known to be contained in the interval [(bandIndex-1)pi, bandIndex*pi].
	Inputs:
		bandIndex: 	int, index of band to estimate start of
		alpha: 	float, value of the vertex-scaling parameter alpha
		nPts: 	(optional) int - default 1000, number of points to subdivide the interval [(bandIndex-1)pi, bandIndex*pi] into to attempt to estimate the start point of the band
		precise: (optional) bool - default False, if True then nPts will be used to estimate a start point of the band, then we will call scipy's fsolve to get a more precise estimate for the start of the band
	Outputs:
		bandStart: 	float, the estimated start point of the band I_bandIndex
	'''
	
	if ((alpha<=2) and (bandIndex==1)):
		#if alpha<=2 the first band is simply (0,pi), so return 0.
		return 0.
	
	#otherwise, compute an estimate for the start of the band
	wTestPts = linspace((bandIndex-1)*pi, bandIndex*pi, num=nPts, endpoint=True)
	wTestPts = wTestPts[1:] #we know that (bandIndex-1)pi is part of the previous band, so ignore it
	
	LHSVals = LHS(wTestPts, alpha) #evaluate LHS
	wBandLowest = wTestPts[np.abs(LHSVals)<=1][0] #any w corresponding to an absolute value of the LHS that's less than one is a part of this band. NOTE: np.abs(LHSVals)<=1 should give an array that is just False False... False True True True.... True
	wOutBoundHighest = wTestPts[np.abs(LHSVals)>1][-1] #this is the greatest value of w we looked at that is not part of the band
	
	if precise:
		#if true, we want a better estimate for the start of the band, use fsolve to get it
		x0 = (wBandLowest + wOutBoundHighest)/2 #start point for fsolve
		if bandIndex%2==0:
			#if bandIndex is even, then LHS(bandIndex*pi) = 1, so at the other end of the band we will have LHS(bandStart)=-1, so let's search for that point, which we know is in the interval [wOutBoundHighest, wBandLowest]
			#create lambda function for fsolve. fsolve solves func(x)=0, so we need to take the -1 onto the LHS. The derivative remains unchanged
			f = lambda w: LHS(w, alpha) + 1
			df = lambda w: LHSDeriv(w, alpha)
			bandStart = fsolve(f, x0, fprime=df)
		else:
			#if bandIndex is odd, then LHS(bandIndex*pi) = -1, so at the other end of the band we will have LHS(bandStart)=1
			#create lambda function for fsolve. fsolve solves func(x)=0, so we need to take the +1 onto the LHS. The derivative remains unchanged
			f = lambda w: LHS(w, alpha) - 1
			df = lambda w: LHSDeriv(w, alpha)
			bandStart = fsolve(f, x0, fprime=df)
		if (bandStart>wBandLowest or bandStart<wOutBoundHighest):
			#if the start of the band is found to be outside the region we just determined it was in, print a warning and default to the safe estimate of wOutBoundHighest
			warn('Precise value for start of band %d out of estimated range (%.3f, %.3f) - using imprecise estimate instead' % (bandIndex, wOutBoundHighest, wBandLowest))
			bandStart = wOutBoundHighest
	else:
		#if false, we will just use wOutBoundHighest as an estimate for the start point, as we defintely contain the band within this region
		bandStart = wOutBoundHighest
	return bandStart

def BandIDoS(bandIndex, alpha, nPts=1000, **kwargs):
	'''
	For a given band I_bandIndex, compute the Integrated Density of States (IDos) for that band. This is a function of x, for x in the range [(bandIndex-1)pi, (bandIndex)pi], that returns values between 0 and (2pi)^2.
	INPUTS:
		bandIndex: 	int, index of band to compute IDoS of
		alpha: 	float, value of the vertex-scaling parameter alpha
		nPts: 	(optional) int - default 1000, number of points to uniformly mesh each component of the quasimomentum into, unless custom values are provided
	**kwargs:
		t1: 	(optional) (n,) float numpy array, custom points to use for discretising the 1st component of the quasimomentum
		t2: 	(optional) (n,) float numpy array, custom points to use for discretising the 2ns component of the quasimomentum
		visual: 	(optional) bool, if provided as True then the IDoS will be plotted before the function returns
		kwargs will be passed to the following functions calls: 
			Visual_BandIDoS
	OUTPUTS:
		bIDoS: function, the IDoS for this band - see the function's docstring via help() for details.	
	'''
	
	#check if custom meshes for theta have been provided
	if 't1' in kwargs.keys():
		#custom theta1 range provided, use this
		theta1 = kwargs['t1']
		t1Pts = theta1.shape[0]
	else:
		#use linspace for theta1, with nPts points
		theta1 = linspace(-pi, pi, num=nPts, endpoint=True)
		t1Pts = nPts
	if 't2' in kwargs.keys():
		#custom theta2 range provided, use this
		theta2 = kwargs['t1']
		t2Pts = theta2.shape[0]
	else:
		#use linspace for theta2, with nPts points
		theta2 = linspace(-pi, pi, num=nPts, endpoint=True)
		t2Pts = nPts
		
	#compute midpoint values for each component of the quasimomentum
	t1Mids = (theta1[1:] + theta1[:-1]) / 2.
	t2Mids = (theta2[1:] + theta2[:-1]) / 2.
	#compute differences between consecutive values for each component of QM
	t1Diffs = (theta1[1:] - theta1[:-1])
	t2Diffs = (theta2[1:] - theta2[:-1])
	#compute band range, so that we can get a first estimate for fsolve that lies in the band
	wEnd = bandIndex*pi
	wStartEst = EstimateBandStart(bandIndex, alpha, precise=False)
	w0 = (wStartEst + wEnd)/2.
	#create storage matrices...
	wBandAreas = np.outer(t1Diffs, t2Diffs) #outer product to get the 'cell areas'. wBandAreas[i,j] = (theta1[i]-theta1[i-1])*(theta2[i]-theta2[i-1])
	#super-inefficient fsolve to get the w values corresponding to the theta midpoints. Need to do an fsolve for each mid-mesh point to get the values of w each QM-value corresponds to, in this band
	wBandVals = np.zeros_like(wBandAreas)
	df = lambda x: LHSDeriv(x, alpha) #function handle for derivative is the same for all the cases
	print('Solving for w for each theta pair, this may take some time! Proportion complete:')
	for i in range(t1Pts-1):
		print('%.3f,' % (i/(t1Pts-1)), end=' ')
		for j in range(t2Pts-1):
			theta = np.asarray([t1Mids[i], t2Mids[j]])
			RHSVal = ProductCos(theta)
			f = lambda x: LHS(x, alpha) - RHSVal
			wBandVals[i,j] = fsolve(f, w0, fprime=df)
	print('Complete')
	
	#check that all the values that fsolve found are still in this band, otherwise we have a problem!
	outOfBounds = (wBandVals<wStartEst) & (wBandVals>wEnd)
	if outOfBounds.sum()!=0:
		#there was at least one value that was found, and doesn't lie in the band - throw warning and exit
		warn('At least one theta pair has not been matched to an eigenvalue in this band.')
		print('Values outside band I_%d [%.3f, %.3f]:' % (bandIndex, wStartEst, wEnd))
		for i in range(outOfBounds.shape[0]):
			for j in range(outOfBounds.shape[1]):
				if outOfBounds[i,j]:
					#if true, this was a bad point
					print('w: %.3f, t1: %.3f, t2: %.3f' % (wBandVals[i,j], t1Mids[i], t2Mids[j]))
		print('Provided output: wBandVals, t1Mids, t2Mids, outOfBounds')
		return wBandVals, t1Mids, t2Mids, outOfBounds
	
	#otherwise, we have found the values of w (in this band) that each QM-value corresponds to, now we just write a function that evaluates the IDoS for this band!
	def bIDoS(x):
		'''
		Given a value x>=0, returns the Lebesgue measure of the region {theta | theta corresponds to an eigenvalue w in the range (0,x) intersected with I_bandIndex}. Note that this is the Integrated Density of States (IDoS) for a single band, so takes values in the range [0, (2pi)^2], and is 0 for x lower than the band start and (2pi)^2 for x greater than the band end.
		INPUTS:
			x: 	(n,) float numpy array, vector of evaluation variables
		OUTPUTS:
			iDoS: (n,) float, values of the integrated density of states for this band
		'''
		
		#this is just so that the function can be used as if it was vectorised
		if np.isscalar(x):
			iDoS = np.sum(wBandAreas[wBandVals<x])
		else:
			iDoS = np.zeros_like(x, dtype=float)
			for i in range(np.shape(x)[0]):
				iDoS[i] = np.sum(wBandAreas[wBandVals<x[i]])
		return iDoS
	#with the function created, we can now return it. But first we check if the user wanted to see a plot of this thing too
	
	if 'visual' in kwargs.keys():
		if kwargs['visual']:
			f = Visual_BandIDoS(bIDoS, bandIndex, bandBounds=np.asarray([wStartEst, wEnd]), **kwargs)
			f.show()
	return bIDoS

def BandDoS(bandIDoS, dx=1e-4, **kwargs):
	'''
	Given the integrated density of states for a given band, compute estimates to the density of states itself using finite differences.
	Note that this is a primitive method, more sophisticated methods exist (which are in general much more stable) and should be considered if the result of this function is poor or displayed unexpected behavour.
	INPUTS:
		bandIDoS: 	lambda function, which takes in a single arugment x and returns the IDoS for a certain band
		dx: 	(optional) float - default 1e-4, the step size to use in finite difference code
		bandIndex: (optional) int - default 1, if plotting is 
	OUTPUTS:
		DoS: 	lambda function, takes a single argument x and returns the DoS for the same band as bandIDoS. See function docstring for further details.
		
	The argument dx needs to be given some thought - due to the way in which we construct the IDoS (by gridding the QM and testing the midpoint of the resulting cells in the mesh) the level of refinement in our QM mesh places a limit on the value of dx that we can use in estimating the derivative. A mesh size dt in QM-space will, through equation (2.7), result in some quantifiable expected change dw in the eigenvalue, but this is a jump and not a smooth transition in the eyes of the machine. As a result, IDoS is actually a series of tiny steps rather than a smooth line (it's just that the mesh size dt is small enough to make it look smooth) and hence DoS will display "spikes" if dx<dw (as IDoS is constant over intervals <dw in length).
	As such, care should be taken when choosing dx, with this consideration in mind.
	'''
	
	def bDoS(x):
		'''
		Given a value x>=0, returns the Density of States (DoS) for a single band, computed by finite differences on the corresponding Integrated Density of States
		INPUTS:
			x: 	(n,) float numpy array, vector of evaluation variables
		OUTPUTS:
			iDoS: (n,) float, values of the integrated density of states for this band
		'''
		
		DoS = ( bandIDoS(x+dx) - bandIDoS(x) ) / dx
		return DoS

	return bDoS

##Begin test and visual functions

def Visual_LHS(alpha, firstBand=1, lastBand=2, bandPts=1000, onePlot=True):
	'''
	Function to visualise the LHS of (2.7) and spectral bands on a single plot.
	INPUTS:
		alpha: 	float, value of the vertex-scaling parameter alpha
		firstBand: 	(optional) int - default 1, the first band from which to begin the plot
		lastBand: 	(optional) int - default 2, the last band to include in the plot
		bandPts: 	(optional) int - default 1000, number of points to uniformly mesh each band into when plotting
		onePlot: 	(optional) bool - default False, if True then the output will consist of one figure with two subplots, one for the LHS and the other for the spectrum. If False, the two plots will share an x-axis
	OUTPUTS:
		f: 	figure handle, figure displaying the spectral bands and LHS of (2.7), see figure 2.4 in the TFR.
	Note: One can retrieve the x,y data from the figure handle f, see https://stackoverflow.com/questions/20130768/retrieve-xy-data-from-matplotlib-figure
	'''
	
	if firstBand>lastBand:
		raise ValueError('firstBand > lastBand (checked %d > %d) - aborting plot.' % (firstBand, lastBand))
	elif firstBand<=0:
		raise ValueError('firstBand <=0 (got %d)' % (firstBand))
	else:
		nBands = lastBand-firstBand +1 #if equal, we still want to plot that one band

	wPts = linspace((firstBand-1)*pi, lastBand*pi, num=bandPts*nBands, endpoint=True)
	LHSVals = LHS(wPts, alpha)
	specPts = np.ones_like(LHSVals, dtype=int)
	specPts[np.abs(LHSVals)<=1] = 0 #these are the points that lie in the spectrum, but we want them plotted at 0, hence 0=good and 1=bad in this case
	specPts[0] = 1 #LHS is only valid for positive real line, and wPts[0] is always omega=0, so manually remove this point

	if onePlot:
		f, ax = plt.subplots()
		ax.set_title(r'$\alpha=%.3f$' % (alpha))
		ax.set_xlabel(r'$\frac{\omega}{\pi}$')
		ax.set_ylabel(r'Spectral Bands, and $\Xi$')
		ax.plot(wPts/pi, LHSVals, 'r')
		ax.plot(wPts/pi, np.ones_like(LHSVals, dtype=int), 'k')
		ax.plot(wPts/pi, -1*np.ones_like(LHSVals, dtype=int), 'k')
		ax.plot(wPts[specPts==0]/pi, specPts[specPts==0], 'bx')
	else:
		f, ax = plt.subplots(1,2, gridspec_kw={'wspace': 0.5})
		ax[0].set_title(r'$\alpha=%.3f$' % (alpha))
		ax[1].set_title(r'$\alpha=%.3f$' % (alpha))
		ax[0].set_xlabel(r'$\frac{\omega}{\pi}$')
		ax[0].set_ylabel(r'$\Xi$')
		ax[1].set_xlabel(r'$\frac{\omega}{\pi}$')
		ax[1].set_ylabel(r'Spectral Bands')		
		ax[0].plot(wPts/pi, LHSVals, 'r')
		ax[0].plot(wPts/pi, np.ones_like(LHSVals, dtype=int), 'k')
		ax[0].plot(wPts/pi, -1*np.ones_like(LHSVals, dtype=int), 'k')
		ax[1].plot(wPts[specPts==0]/pi, specPts[specPts==0], 'bx')
	
	return f

def Test_BandEstimatesVisual(bandIndex, alpha, nPts=1000):
	'''
	Test function, to visually check that the estimate for the start point of the band with index bandIndex is accurate
	INPUTS:
		bandIndex: 	int, index of band to estimate start of
		alpha: 	float, value of the vertex-scaling parameter alpha
		nPts: 	(optional) int - default 1000, number of points to subdivide the interval [(bandIndex-1)pi, bandIndex*pi] into to attempt to estimate the start point of the band
	OUTPUTS:
		f: 	figure, figure containing the visualisation
	'''
	
	wTestPts = linspace((bandIndex-1)*pi, bandIndex*pi, num=5*nPts, endpoint=True) #extra points for precision
	LHSVals = LHS(wTestPts, alpha) #evaluate LHS at all points in the interval
	
	#get estimates for the start of the band, both using the precise and imprecise methods
	estBandStart = EstimateBandStart(bandIndex, alpha, nPts=nPts)
	estBandStartPrecise = EstimateBandStart(bandIndex, alpha, nPts=nPts, precise=True)
	
	f, ax = plt.subplots()
	ax.plot(wTestPts/pi, LHSVals, 'red')
	ax.plot(wTestPts/pi, np.ones_like(wTestPts), 'black')
	ax.plot(wTestPts/pi, -np.ones_like(wTestPts), 'black')
	ax.plot(estBandStart/pi, 1, 'xg')
	ax.plot(estBandStart/pi, -1, 'xg')
	ax.plot(estBandStartPrecise/pi, 1, 'xb')
	ax.plot(estBandStartPrecise/pi, -1, 'xb')
	ax.set_ylim(-1.5,1.5)
	ax.set_xlabel(r'$\frac{\omega}{\pi}$')
	ax.set_title(r'Estimate Start Points for band $I_{%d}$' %bandIndex)
	
	return f

def Visual_BandIDoS(d, bandIndex, **kwargs):
	'''
	Function to visualise the numerically estimated integrated density of states (IDoS) produced by the function BandIDoS.
	INPUTS:
		d: 	lambda function of one variable, the (successful) output of BandIDoS
		bandIndex: 	int, the index of the band that d is the IDoS for
	**kwargs:
		nPts: 	(optional) int - default 1000, number of meshpoints to use in plot
		bandBounds: 	(2,) float numpy array, if provided it should be the start and endpoints (in that order) of the band which will be marked on the output
	OUTPUTS:
		f: 	figure, figure containing a plot of the IDoS
	'''
	
	if 'nPts' in kwargs.keys():
		nPts = kwargs['nPts']
	else:
		nPts = 1000
	if 'bandBounds' in kwargs.keys():
		wStart = kwargs['bandBounds'][0]
		wEnd = kwargs['bandBounds'][1]
		plotBB = True
	else:
		plotBB = False
	
	if plotBB:
		wBandPts = linspace(wStart, wEnd, num=nPts, endpoint=True)
		wPts = np.concatenate((linspace((bandIndex-1)*pi, wStart, endpoint=False), wBandPts, linspace(wEnd, wEnd+pi/8)))  #add some padding around the band just to check we are constant outside it
		bandIDoSVals = d(wPts)
		
		f, ax = plt.subplots()
		ax.plot(wPts/pi, bandIDoSVals, 'red')
		ax.plot(wStart/pi, d(wStart), 'xb')
		ax.plot(wEnd/pi, d(wEnd), 'xb')
		ax.set_xlabel(r'$\frac{\omega}{\pi}$')
		ax.set_ylabel(r'IDoS')
		ax.set_title(r'Integrated Density of States for band $I_{%d}$' %bandIndex)
	else:
		wPts = linspace((bandIndex-1)*pi - pi/2, bandIndex*pi + pi/2, num=2*nPts, endpoint=True) #add some padding around the band just to check we are constant outside it
		bandIDoSVals = d(wPts)
		
		f, ax = plt.subplots()
		ax.plot(wPts/pi, bandIDoSVals, 'red')
		ax.set_xlabel(r'$\frac{\omega}{\pi}$')
		ax.set_ylabel(r'IDoS')
		ax.set_title(r'Integrated Density of States for band $I_{%d}$' %bandIndex)		
	return f

def Visual_BandDoS(d, bandIndex, **kwargs):
	'''
	Function to visualise the numerically estimated density of states (DoS) produced by the function BandDoS.
	INPUTS:
		d: 	lambda function of one variable, the (successful) output of BandDoS
		bandIndex: 	int, the index of the band that d is the DoS for
	**kwargs:
		nPts: 	(optional) int - default 1000, number of meshpoints to use in plot
		bandBounds: 	(2,) float numpy array, if provided it should be the start and endpoints (in that order) of the band which will be marked on the output
	OUTPUTS:
		f: 	figure, figure containing a plot of the IDoS
	'''
	
	if 'nPts' in kwargs.keys():
		nPts = kwargs['nPts']
	else:
		nPts = 1000
	if 'bandBounds' in kwargs.keys():
		wStart = kwargs['bandBounds'][0]
		wEnd = kwargs['bandBounds'][1]
		plotBB = True
	else:
		plotBB = False
	
	if plotBB:
		wBandPts = linspace(wStart, wEnd, num=nPts, endpoint=True)
		wPts = np.concatenate((linspace((bandIndex-1)*pi, wStart, endpoint=False), wBandPts, linspace(wEnd, wEnd+pi/8)))  #add some padding around the band just to check we are constant outside it
		bandDoSVals = d(wPts)
		
		f, ax = plt.subplots()
		ax.plot(wPts/pi, bandDoSVals, 'red')
		ax.plot(wStart/pi, d(wStart), 'xb')
		ax.plot(wEnd/pi, d(wEnd), 'xb')
		ax.set_xlabel(r'$\frac{\omega}{\pi}$')
		ax.set_ylabel(r'DoS')
		ax.set_title(r'IDensity of States for band $I_{%d}$' %bandIndex)
	else:
		wPts = linspace((bandIndex-1)*pi - pi/2, bandIndex*pi + pi/2, num=2*nPts, endpoint=True) #add some padding around the band just to check we are constant outside it
		bandDoSVals = d(wPts)
		
		f, ax = plt.subplots()
		ax.plot(wPts/pi, bandDoSVals, 'red')
		ax.set_xlabel(r'$\frac{\omega}{\pi}$')
		ax.set_ylabel(r'DoS')
		ax.set_title(r'Density of States for band $I_{%d}$' %bandIndex)		
	return f

def Visual_FullDoS(alpha, firstBand=1, lastBand=2, bandPts=1000, normIDoS=True, **kwargs):
	'''
	Function to visualise the Integrated Density of States (IDoS), Density of States (DoS), and spectral bands on a single plot consisting of subfigures.
	INPUTS:
		alpha: 	float, value of the vertex-scaling parameter alpha
		firstBand: 	(optional) int - default 1, the first band from which to begin the plot
		lastBand: 	(optional) int - default 2, the last band to include in the plot
		bandPts: 	(optional) int - default 1000, number of points to uniformly mesh each band into when plotting
		normIDoS: 	(optional) bool - default False, if True then the IDoS will be plotted for each band, rather than the running total IDoS of all the bands. Essentially upon reaching the endpoint of a band, IDoS has (2pi)^2 subtracted from it, resetting it's value to 0 before starting the next band.
	**kwargs:
		t1: 	(n,) float numpy array, custom points to use for discretising the 1st component of the quasimomentum
		t2: 	(n,) float numpy array, custom points to use for discretising the 2ns component of the quasimomentum
		nPts: 	number of points to uniformly mesh each component of the quasimomentum into, unless custom values are provided
		dx: 	float, the step size to use in finite difference code for computing DoS from IDoS, see BandDoS function for more details
	OUTPUTS:
		f: 	figure handle, figure displaying the DoS, IDoS, and spectral bands stacked atop each other
	Note: One can retrieve the x,y data from the figure handle f, see https://stackoverflow.com/questions/20130768/retrieve-xy-data-from-matplotlib-figure
	'''
	
	if firstBand>lastBand:
		raise ValueError('firstBand > lastBand (checked %d > %d) - aborting plot.' % (firstBand, lastBand))
	elif firstBand<=0:
		raise ValueError('firstBand <=0 (got %d)' % (firstBand))
	else:
		nBands = lastBand-firstBand +1 #if equal, we still want to plot that one band
	
	#catch potential mesh-size problems
	if ('nPts' in kwargs.keys()) and ('dx' in kwargs.keys()):
		if kwargs['dx'] < 1./kwargs['nPts']:
			warn('dx (%.1e) is less than QM mesh size (%.1e); will continue, but output will be inaccurate.' % (kwargs['dx'], 1./kwargs['nPts']))
	elif 'nPts' in kwargs.keys():
		if kwargs['nPts']<100:
			warn('QM mesh size smaller than finite difference step used in calculation, consider revising (nPts=%d)' % (kwargs['nPts']))
	elif  'dx' in kwargs.keys():
		if kwargs['dx']<1e-3:
			warn('Finite difference step greater than QM mesh size, consider revising (dx=%.1e)' % (kwargs['dx']))

	#we will create a figure consisting of 3 subfigures
	f, ax = plt.subplots(3, sharex=True, gridspec_kw={'hspace': 0.5})
	ax[2].set_xlabel(r'$\frac{\omega}{\pi}$')
	#the same mesh will be used for each plot...
	wPts = linspace((firstBand-1)*pi, lastBand*pi, num=bandPts*nBands, endpoint=True)
	#The middle figure will display the band structure
	LHSAbs = np.abs(LHS(wPts, alpha))
	bandIndicator = np.ones_like(LHSAbs, dtype=int)
	bandIndicator[LHSAbs>1] = 0
	bandIndicator[0] = 0 #wPts[0]=0, but this isn't a point in the spectrum
	ax[1].plot(wPts/pi, bandIndicator, 'rx')
	ax[1].set_ylabel(r'Spectral Bands')
	ax[1].set_ylim(0.9,1.1)
	#The top figure will be the integrated DoS, and the bottom the regular DoS
	iDoSVals = np.zeros_like(wPts, dtype=float)
	DoSVals = np.zeros_like(wPts, dtype=float)
	for band in range(firstBand, lastBand+1):
		print('Working on band: %d / %d' % (band, lastBand))
		#for each band, we need to generate the IDoS and DoS, then add them to the running total that we have going
		bandIDoS = BandIDoS(band, alpha, **kwargs)
		bandDoS = BandDoS(bandIDoS, **kwargs)
		iDoSVals += bandIDoS(wPts)
		DoSVals += bandDoS(wPts)
	if normIDoS:
		#if normIDoS is true, we want to subtract (2pi)^2 from IDoS after each band (so the plot displays the relative increase of each band, and doesn't get a large y-axis)
		iDoSVals -= ((2*pi)**2)*(firstBand-1) #starting at firstBand, there are firstBand-1 lower bands which are all adding an extra (2pi)^2 to the IDoS 
		for band in range(firstBand, lastBand): #don't need to do the last band as it should have been normalised
			#now for each band that we go through, we need to sucessively remove the extra (2pi)^2 contribution it gives
			subtractArray = np.zeros_like(wPts, dtype=int)
			subtractArray[wPts>(band*pi)] = 1
			iDoSVals -= ((2*pi)**2)*subtractArray #take away the (2pi)^2 contribution this band has on all those above it
		ax[0].set_ylabel(r'$\frac{\mathrm{IDoS \ (rel.)}}{(2\pi)^2}$')
	else:
		ax[0].set_ylabel(r'$\frac{\mathrm{IDoS}}{(2\pi)^2}$')
	iDoSVals /= (2*pi)**2

	#now we place these values into the figure
	ax[0].plot(wPts/pi, iDoSVals, 'b')
	ax[2].plot(wPts/pi, DoSVals, 'b')
	ax[2].set_ylabel(r'DoS')
	ax[0].set_title(r'$\alpha=%.2f$' % (alpha))
	
	return f