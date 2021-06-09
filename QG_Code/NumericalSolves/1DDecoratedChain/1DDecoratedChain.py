#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 08:38:23 2021

@author: will

Numerics for the 1D-chain graph with a decoration example.
The determinant to be considered is
\det\tilde{M} = 2 w^3 (s_a s_b s_2)^2 c_2 * [cos(theta)+0.5-1.5cos(w)+w\alpha*sin(w)/2]

with 
H^2 = s_a s_b s_2,
s_a = sin(a*w),
s_b = sin(b*w), b = 1-a,
s_2 = sin(w/2)
c_2 = cos(w/2)

No explicit formula for the eigenvalue branches is available.
"""

import numpy as np
from numpy import sin, cos, exp

import matplotlib.pyplot as plt
from matplotlib import rc   
rc('text', usetex=True)

from scipy.optimize import fsolve

def DispRel(w, theta, alpha, alt=False):
	'''
	The "actually useful" dispersion relation part of the determinant;
	
	cos(theta) + 0.5 - 1.5cos(w) + w*alpha*sin(w)/2.
	
	If alt=True, then we return the form of the DR that can can be used to check whether
	a given w satisfies the DR for some value of theta,
	3cos(w) - w*alpha*sin(w) +1.
	
	See DROmegaSolve for it's utility.
	'''
	
	if alt:
		return 3.*cos(w) - w*alpha*sin(w) + 1.
	else:
		return cos(theta) + 0.5 - (1.5*cos(w)) + ((w*alpha*sin(w))/2.)

def detM(w, theta, alpha, a):
	'''
	Evaluates the analytic expression we have obtained for the determinant of \tilde{M},
	\det\tilde{M} = 2 w^3 (s_a s_b s_2)^2 c_2 * [cos(theta)+0.5-1.5cos(w)+w\alpha*sin(w)/2]
	'''
	
	b = 1. - a
	sa = sin(a*w)
	sb = sin(b*w)
	s2 = sin(w/2.)
	c2 = cos(w/2.)
	
	bracsTerm = DispRel(w, theta, alpha)
	detVal = 2. * w*w*w * sa*sa * sb*sb * s2*s2 * c2 * bracsTerm
	return detVal

def H2(w, a):
	'''
	Evaluates the function H^2, where
	H^{(2)}(w) = s_a * s_b * s_2.
	'''
	
	b = 1. - a
	return sin(a*w) * sin(b*w) * sin(w/2.)

def DROmegaSatisfy(w, alpha):
	'''
	We can see that DispRel(w, theta, alpha)=0 for _some_ value of theta when
	|3cos(w) - w*alpha*sin(w) +1| <= 2.
	This function performs this check for each value in w (at the fixed value alpha),
	returning a Treu/False array of whether this condition is satisfied
	'''
	
	testVals = np.abs( 3.*cos(w) - w*alpha*sin(w) + 1. )
	return testVals <= 2.

def H2Roots(wMin, wMax, a):
	'''
	Compute the roots of the function H2 in the interval [wMin, wMax].
	We know the roots of H2 have the form
	w = n*pi/a, w = n*pi/b, w = n*pi/2,
	so it is just a case of computing all of them explictly.
	
	Return 3 arrays containing the roots, 
	the first the roots of the form n*pi/a,
	the second with the roots of the form n*pi/b,
	and the third with roots of the from n*pi/2.
	'''
	
	aMinN = np.ceil(wMin*a/np.pi)
	aMaxN = np.floor(wMax*a/np.pi)
	aRoots = np.arange(aMinN, aMaxN+1., dtype=float) * np.pi/a
	
	b = 1. - a
	bMinN = np.ceil(wMin*b/np.pi)
	bMaxN = np.floor(wMax*b/np.pi)
	bRoots = np.arange(bMinN, bMaxN+1., dtype=float) * np.pi/b
	
	minN2 = np.ceil(wMin/(2.*np.pi))
	maxN2 = np.floor(wMax/(2.*np.pi))
	roots2 = np.arange(minN2, maxN2+1., dtype=float) * np.pi * 2.
	
	return aRoots, bRoots, roots2

def TildeM(w, theta, alpha, a, beta = np.pi/4.):
	'''
	Assembles the matrix \tilde{M}, which is a 3*3 matrix.
	If w is of shape (n,) rather than just a scalar, the function returns a
	(3,3,n) array with the matrix \tilde{M} at each value of w given
	'''

	sa = sin(a*w)
	ca = cos(a*w)	
	b = 1. - a
	sb = sin(b*w)
	cb = cos(b*w)
	s2 = sin(w/2.)
	c2 = cos(w/2.)
	gamma = cos(beta)/2.
	
	if np.ndim(w)==0:
		# scalar w, return 3*3 matrix
		M = np.zeros((3,3), dtype=complex)
		M[0,0] = w*w*alpha*sa*sb*s2 - w*ca*sb*s2 - w*sa*cb*s2 - w*sa*sb*c2
		M[0,1] = w*exp(1.j * gamma)*sa*sb
		M[0,2] = w*exp(1.j*a*theta)*sb*s2 + w*exp(-1.j*b*theta)*sa*s2
		M[1,0] = w*exp(-1.j * gamma)*sa*sb
		M[1,1] = -w*sa*sb*c2
		M[2,0] = w*exp(-1.j*a*theta)*sb*s2 + w*exp(1.j*b*theta)*sa*s2
		M[2,2] = -w*ca*sb*s2 - w*sa*cb*s2
	else:
		# vector w, return 3*3*n array
		M = np.zeros((3,3,np.shape(w)[0]), dtype=complex)
		M[0,0,:] = w*w*alpha*sa*sb*s2 - w*ca*sb*s2 - w*sa*cb*s2 - w*sa*sb*c2
		M[0,1,:] = w*exp(1.j * gamma)*sa*sb
		M[0,2,:] = w*exp(1.j*a*theta)*sb*s2 + w*exp(-1.j*b*theta)*sa*s2
		M[1,0,:] = w*exp(-1.j * gamma)*sa*sb
		M[1,1,:] = -w*sa*sb*c2
		M[2,0,:] = w*exp(-1.j*a*theta)*sb*s2 + w*exp(1.j*b*theta)*sa*s2
		M[2,2,:] = -w*ca*sb*s2 - w*sa*cb*s2
	return M

def SolveForTheta(w, alpha, t0=0.0):
	'''
	Given that the point w satisfies DROmegaSatisfy, determine the value of theta which
	corresponds to this solution.
	'''
	
	f = lambda t: DispRel(w, t, alpha, alt=False) #function handle to solve for theta
	tSolve = fsolve(f, t0)
	#fit to QM range
	while tSolve >= np.pi:
		tSolve -= 2*np.pi
	while tSolve < -np.pi:
		tSolve += 2*np.pi
	return tSolve

def ExploreBranches(w0, alpha, a, beta, wWidth=.5, nPts=501, showFig=True, findTheta=True, tManual=0.):
	'''
	Given that the point w0 satisfies DROmegaSatisfy, we want to examine the behaviour of the
	eigenvalue branches of the matrix \tilde{M} around that point.
	This function aims to do this graphically:
		- first by determining the theta value t0 for which w0 is a solution
		- computing the eigenvalues of \tilde{M} at t0 in the range w0 +- wWidth
		- plotting the resulting eigenvalue branches
	'''
	
	if findTheta==True:
		#determine theta value for which w0 is a solution
		t0 = SolveForTheta(w0, alpha)
	else:
		#use user-input value of theta
		t0 = tManual
	#create range for branches to be plotted in.
	#NOTE: may wish to avoid including w0 in this array! 
	wRange = np.linspace(w0 - wWidth, w0 + wWidth, nPts)
	#compute M-matrix at each value in wRange
	tildeM = TildeM(wRange, t0, alpha, a, beta = beta)
	nEvals = np.shape(tildeM)[0]
	#compute eigenvalues at each value in wRange...
	eVals = np.zeros((nEvals,nPts), dtype=float)
	#this loop may take some time
	#also, the eigenvalues come back sorted, so we might have problems with
	#branches intersecting in the vicinity of w0
	for i in range(nPts):
		eVals[:,i] = np.linalg.eigvalsh(tildeM[:,:,i])
	#we should now be able to plot eVals[j,:] against wRange to view the eigenvalue branches
	if showFig:
		fig, ax = plt.subplots(1)
		ax.set_title(r'Eigenvalue branches near $\omega_0=%.3f \pi$, $\alpha=%.3f$, $\theta=%.3f$' % (w0/np.pi, alpha, theta))
		ax.set_xlabel(r'$\frac{\omega}{\pi}$')
		for i in range(nEvals):
			ax.plot(wRange/np.pi, eVals[i,:], 'k')
		ax.scatter(w0/np.pi, 0.0, c='r', marker='o', s=10)
		fig.show()
	
	return wRange, eVals

def ExtractBranch(eVals, branchTags):
	'''
	Given the array eVals (from ExploreBranchLimit), which is of shape (nEvals, n),
	extract the value at each point eVals[branchTags, :]
	branchTags should be of shape (n,), and contain the indices of the branch values that we want to extract
	'''
	
	nEvals = np.shape(eVals)[1]
	return eVals[branchTags, np.arange(nEvals)]

nPi = 4
ptsPerPi = 1000

theta = 0.
alpha = 0.25 #alpha >0 since B = -(alpha*w^2) should be used
beta = np.pi/4. #should be irrelevant though
a = 1./np.sqrt(2.)

wRange = np.linspace(0., nPi*np.pi, ptsPerPi*nPi)
drVals = DispRel(wRange, theta, alpha, alt=True)
eVals = DROmegaSatisfy(wRange, alpha)
H2Vals = H2(wRange, a)
aR, bR, r2 = H2Roots(wRange[0], wRange[-1], a)
h2Roots = np.append(r2, np.append(bR, aR))

if False:
	fig, ax = plt.subplots(1)
	ax.set_title(r'Dispersion Relation and Zeros of $H^{(2)}$, $\alpha=%.3f$' % (alpha))
	ax.set_xlabel(r'$\frac{\omega}{\pi}$')
	ax.plot(wRange/np.pi, drVals, 'b', label=r'Dispersion Relation')
	ax.axhline(2., c='k')
	ax.axhline(-2., c='k')
	ax.set_xlim(0, wRange[-1]/np.pi)
	ax.scatter(wRange[eVals]/np.pi, np.zeros_like(wRange[eVals]), c='r', marker='x', s=3, label=r'Solutions to DR')
	ax.scatter(h2Roots/np.pi, np.zeros_like(h2Roots), c='g', marker='x', s=3, label=r'$H^{(2)}(\omega)=0$')
	#ax.scatter(aR/np.pi, np.zeros_like(aR), facecolors='none', edgecolors='g', marker='x', s=3, label=r'$H^{(2)}(\omega)=0$')
	ax.legend()
	saveStr = 'DR+H2-alpha%.3f.pdf' % (alpha)
	fig.savefig(saveStr, bbox_inches='tight')
	fig.show()

#### for computing eigenvalues, perhaps we need to be careful?
if False:
	print('Now making eigenvalue plots, subject to eyeball norm')
	nPi = 4
	ptsPerPi = 1000
	
	theta = 0.
	alpha1 = 1.
	alpha2 = 0.25
	beta = np.pi/4. #should be irrelevant though
	a = 1./np.sqrt(2.)
	
	wRange = np.linspace(0., nPi*np.pi, ptsPerPi*nPi)
	w0 = aR[1]
	
	#branches for alpha=1.0
	wR, eVals1 = ExploreBranches(w0, alpha1, a, beta, wWidth=0.25*np.pi, nPts=1000, showFig=False)
	tags11 = np.zeros_like(wR, dtype=int)
	tags11[wR > w0] = 1
	tags12 = np.zeros_like(wR, dtype=int)
	tags12[wR <= w0] = 1
	b11 = ExtractBranch(eVals1, tags11)
	b12 = ExtractBranch(eVals1, tags12)
	
	#branches for alpha=0.25
	wR, eVals2 = ExploreBranches(w0, alpha2, a, beta, wWidth=0.25*np.pi, nPts=1000, showFig=False)
	tags21 = np.zeros_like(wR, dtype=int)
	tags21[wR > w0] = 1
	tags22 = np.zeros_like(wR, dtype=int)
	tags22[wR <= w0] = 1
	b21 = ExtractBranch(eVals2, tags21)
	b22 = ExtractBranch(eVals2, tags22)
	
	# h2 at relevant points
	h2 = H2(wR, a)
	
	#one figure, 4 subplots
	eFig, eAx = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False, figsize=(10.,5.))
	# main title
	eFig.suptitle(r'Eigenvalue branches near $\omega_0=%.3f\pi$' % (w0/np.pi))
	# (shared) axis labels, and column titles
	eAx[1,0].set_xlabel(r'$\frac{\omega}{\pi}$')
	eAx[1,0].set_ylabel(r'$\left(H^{(2)}\right)^{-1}\widetilde{\beta}_{j}$')
	eAx[0,0].set_title(r'$\alpha_1=%.3f$' % (alpha1))
	eAx[0,0].set_ylabel(r'$\widetilde{\beta}_{j}$')
	eAx[1,1].set_xlabel(r'$\frac{\omega}{\pi}$')
	eAx[0,1].set_title(r'$\alpha_1=%.3f$' % (alpha2))
	# axis markers for showing where branches are 0 at w0
	# alpha = alpha_1
	eAx[0,0].axvline(w0/np.pi, c='k')
	eAx[0,0].axhline(0., c='k')
	eAx[1,0].axvline(w0/np.pi, c='k')
	eAx[1,0].axhline(0., c='k')
	# alpha = alpha_2
	eAx[0,1].axvline(w0/np.pi, c='k')
	eAx[0,1].axhline(0., c='k')
	eAx[1,1].axvline(w0/np.pi, c='k')
	eAx[1,1].axhline(0., c='k')
	# now the actual plotting data
	# first for alpha_1...
	eAx[0,0].plot(wR/np.pi, b11, 'b')
	eAx[0,0].plot(wR/np.pi, b12, 'r')
	eAx[1,0].plot(wR/np.pi, b11/h2, 'b')
	eAx[1,0].plot(wR/np.pi, b12/h2, 'r')
	# now for the alpha_2 column
	eAx[0,1].plot(wR/np.pi, b21, 'b')
	eAx[0,1].plot(wR/np.pi, b22, 'r')
	eAx[1,1].plot(wR/np.pi, b21/h2, 'b')
	eAx[1,1].plot(wR/np.pi, b22/h2, 'r')
	# axis chopping
	eAx[1,0].set_xlim(np.min(wR)/np.pi, np.max(wR)/np.pi)
	eAx[1,1].set_xlim(np.min(wR)/np.pi, np.max(wR)/np.pi)
	eAx[0,0].set_ylim(-4., 1.)
	eAx[0,1].set_ylim(-4., 1.)
	eAx[1,0].set_ylim(-1., 10.)
	eAx[1,1].set_ylim(-1., 10.)
	# annotations post-chopping
	eAx[0,0].annotate(r'$\frac{\omega_0}{\pi}$', (w0/np.pi + 0.01, -4 + 0.03), size=15, va='bottom', ha='left')
	eAx[1,0].annotate(r'$\frac{\omega_0}{\pi}$', (w0/np.pi - 0.01, 10 - 0.5), size=15, va='top', ha='right')
	eAx[0,1].annotate(r'$\frac{\omega_0}{\pi}$', (w0/np.pi + 0.01, -4 + 0.03), size=15, va='bottom', ha='left')
	eAx[1,1].annotate(r'$\frac{\omega_0}{\pi}$', (w0/np.pi - 0.01, 10 - 0.5), size=15, va='top', ha='right')
	#save result before I forget to
	saveStr = 'EvalBranches.pdf'
	eFig.savefig(saveStr, bbox_inches='tight')

# make plots of the eigenvalue branches when qm does (and does not) also solve the bracketed part of the DR
if True:
	print('Making plots to show that theta must be a specfic value for eigenvalue acceptence')
	nPi = 4
	ptsPerPi = 1000
	
	theta = np.pi/2.
	alpha = 0.25
	beta = np.pi/4. #should be irrelevant though
	a = 1./np.sqrt(2.)
	
	wRange = np.linspace(0., nPi*np.pi, ptsPerPi*nPi)
	w0 = aR[1]
	t0 = SolveForTheta(w0, alpha) #ONLY this theta solves the DR
	
	#branches for theta=t0, which solves the bracketed part of the DR as well as being a root of H2
	wR, eValsGood = ExploreBranches(w0, alpha, a, beta, wWidth=0.25*np.pi, nPts=1000, showFig=False)
	tags1Good = np.zeros_like(wR, dtype=int)
	tags1Good[wR > w0] = 1
	tags2Good = np.zeros_like(wR, dtype=int)
	tags2Good[wR <= w0] = 1
	b1Good = ExtractBranch(eValsGood, tags1Good)
	b2Good = ExtractBranch(eValsGood, tags2Good)
	#now get the branches for theta=theta, a manual value which doesn't solve the bracketed term, but is a root of H2
	wR, eValsBad = ExploreBranches(w0, alpha, a, beta, wWidth=0.25*np.pi, nPts=1000, showFig=False, findTheta=False, tManual=theta)
	tags1Bad = np.zeros_like(wR, dtype=int)
	tags1Bad[wR > w0] = 1
	tags2Bad = np.zeros_like(wR, dtype=int)
	tags2Bad[wR <= w0] = 1
	b1Bad = ExtractBranch(eValsBad, tags1Bad)
	b2Bad = ExtractBranch(eValsBad, tags2Bad)
	
	# h2 at relevant points
	h2 = H2(wR, a)
	
	#create plots...
	tFig, tAx = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False, figsize=(10.,5.))
	# main title
	tFig.suptitle(r'Eigenvalue branches near $\omega_0=%.3f\pi, \alpha=%.2f$' % (w0/np.pi, alpha))
	# (shared) axis labels, and column titles
	tAx[1,0].set_xlabel(r'$\frac{\omega}{\pi}$')
	tAx[1,1].set_xlabel(r'$\frac{\omega}{\pi}$')
	tAx[0,0].set_ylabel(r'$\widetilde{\beta}_{j}$')
	tAx[1,0].set_ylabel(r'$\left(H^{(2)}\right)^{-1}\widetilde{\beta}_{j}$')
	tAx[0,0].set_title(r'$\theta=%.3f\pi$' % (t0/np.pi))
	tAx[0,1].set_title(r'$\theta=%.3f\pi$' % (theta/np.pi))
	# axis limits 
	tAx[0,0].set_ylim(-2.,1.)
	tAx[1,0].set_ylim(-2.,10.)
	tAx[0,1].set_ylim(-2.,1.)
	tAx[1,1].set_ylim(-2.,10.)
	tAx[1,0].set_xlim(np.min(wR)/np.pi, np.max(wR)/np.pi)
	tAx[1,0].set_xlim(np.min(wR)/np.pi, np.max(wR)/np.pi)
	# manual hacks/annotations
	tAx[0,0].axhline(y=0.0, c='k')
	tAx[0,0].axvline(x=w0/np.pi, c='k')
	tAx[0,1].axhline(y=0.0, c='k')
	tAx[0,1].axvline(x=w0/np.pi, c='k')
	tAx[1,0].axhline(y=0.0, c='k')
	tAx[1,1].axhline(y=0.0, c='k')
	tAx[1,0].axvline(x=w0/np.pi, c='k')
	tAx[1,1].axvline(x=w0/np.pi, c='k')
	tAx[0,0].annotate(r'$\frac{\omega_0}{\pi}$', (w0/np.pi + 0.01, -2. + 0.03), size=15, va='bottom', ha='left')
	tAx[1,0].annotate(r'$\frac{\omega_0}{\pi}$', (w0/np.pi - 0.01, 10 - 0.5), size=15, va='top', ha='right')
	tAx[0,1].annotate(r'$\frac{\omega_0}{\pi}$', (w0/np.pi + 0.01, -2. + 0.03), size=15, va='bottom', ha='left')
	tAx[1,1].annotate(r'$\frac{\omega_0}{\pi}$', (w0/np.pi - 0.01, 10 - 0.5), size=15, va='top', ha='right')
	# function data
	tAx[0,0].plot(wR/np.pi, b1Good, 'b')
	tAx[0,0].plot(wR/np.pi, b2Good, 'r')
	tAx[0,1].plot(wR/np.pi, b1Bad, 'b')
	tAx[0,1].plot(wR/np.pi, b2Bad, 'r')
	tAx[1,0].plot(wR/np.pi, b1Good/h2, 'b')
	tAx[1,0].plot(wR/np.pi, b2Good/h2, 'r')
	tAx[1,1].plot(wR/np.pi, b1Bad/h2, 'b')
	tAx[1,1].plot(wR/np.pi, b2Bad/h2, 'r')
	#save result before I forget to
	saveStr = 'EvalBranches-Thetas.pdf'
	tFig.savefig(saveStr, bbox_inches='tight')