#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:30:08 2020

@author: will

This file concerns the Thick-Vertex TFR graph, with the Curl-Curl equations posed on it.

NB: We assume the opposite convention to the EKK paper, hence the parameter alpha is negative throughout to be consistent with the TFR results and that paper
"""
#numpy is always needed
import numpy as np
from numpy import sqrt, sin, cos
from numpy import pi
#pandas in case we use dataframes
import pandas as pd
#warnings for plot saving handling
from warnings import warn
#matplotlib for plotting in 3d and 2d
import matplotlib.pyplot as plt
from matplotlib import rc, cm
from mpl_toolkits.mplot3d import axes3d

import seaborn as sns

def FigWindowSettings(axis):
	'''
	Changes a matplotlib.pyplot axis object to have no top and right boarder or tick marks.
	INPUTS:
		axis: 	matplotlib.pyplot axis object
	'''

	# Hide the right and top spines
	axis.spines['right'].set_visible(False)
	axis.spines['top'].set_visible(False)
	# Only show ticks on the left and bottom spines
	axis.yaxis.set_ticks_position('left')
	axis.xaxis.set_ticks_position('bottom')
	return

def GenTitStr(wavenumber, alpha):
	'''
	Generates a title (as a string with latex syntax) for a figure, given the wavenumber and alpha values
	INPUTS:
		wavenumber: 	float, value of the wavenumber to evaluate the DR at
		alpha: 	float, value for the thick-vertex constant alpha
	OUTPUTS:
		titStr: 	string, generated title in latex syntax
	'''
	
	titStr = r'$\frac{\kappa}{\pi}=' + r"{:.2f}".format(wavenumber/pi) + r'$, $\alpha=' + r"{:.2f}".format(alpha) + r'$'
	return titStr

def DispExpr(w, wavenumber, alpha):
	'''
	This is the Dispersion Expression, when it lies between -1 and 1, that omega (w) corresponds to an eigenvalue.
	This is for a fixed wavenumber.
	INPUTS:
		w: 	(n,) float numpy array, values of omega to evaluate the DR at
		wavenumber: 	float, value of the wavenumber to evaluate the DR at
		alpha: 	float, value for the thick-vertex constant alpha
	OUTPUTS:
		dispVals: 	(n,) float numpy array, values of the DispExpr at each omega, wavenumber, alpha
	'''	
	effFreq = sqrt(w**2 - wavenumber**2)
	dispVals = cos(effFreq) - (alpha/4) * (w**2) * sin(effFreq) / effFreq
	
	return dispVals

def FixedWavenumberSlice(wavenumber, alpha, wPts=1000, saveFig=False, saveStr='.pdf'):
	'''
	Creates a plot of the Dispersion Expression for the given value of wavenumber and alpha. Plot can be optionally saved
	INPUTS:
		wavenumber: 	float, value of the wavenumber to evaluate the DR at
		alpha: 	float, value for the thick-vertex constant alpha
		wPts: 	int (default 1000), number of points to plot DR value at
		saveFig: 	bool (default False), if True the plot that is produced will be saved
		saveStr: 	str (default '.pdf'), if saveFig is True then the file produced will be saved with this name
	OUTPUTS:
		fig: 	matplotlib.pyplot.figure object, belonging to the plot that was produced
		ax: 	matplotlib.pyplot.axes object, belonging to the axes on which the DR was plotted
	'''
	
	w = np.linspace(0,6*pi,wPts) + wavenumber
	
	#obtain dispersion relation values
	drVals = DispExpr(w, wavenumber, alpha)
	#create spectral plot by finding points between -1 and +1
	validEvals = abs(drVals)<=1

	#create plot
	fig, ax = plt.subplots(1) #create axes for plotting
	FigWindowSettings(ax) #change figure format to be nice
	ax.plot(w/pi, drVals) #actual dispersion relation
	ax.plot(w/pi, np.ones_like(w), 'k') #line indicating +1
	ax.plot(w/pi, -1*np.ones_like(w), 'k') #line indicating -1
	ax.scatter(w[validEvals]/pi, np.zeros_like(w[validEvals]), s=1, c='r', marker='x')	#red points indicating spectrum
	ax.set_xlabel(r'Frequency, $\frac{\omega}{\pi}$')
	ax.set_ylabel(r'Dispersion Expression Value')
	ax.set_xlim([w[0]/pi, w[wPts-1]/pi])
	ax.set_title(GenTitStr(wavenumber, alpha))
	
	#if we want to save things
	try:
		if saveFig==True and len(saveStr)>4:
			fig.savefig(saveStr)
		elif saveFig:
			raise ValueError('Invalid filename, try something with an extension and at least 2 characters.')
	except ValueError:
		warn('Invalid filename provided, figure not saved!')
	except:
		warn('Unexpected error occurred whilst saving, run in debug mode.')
	
	#if successful, return the plot handles
	return fig, ax

def DispersionPlot(alpha, wPts=1000, kPts=1000, saveFig=False, saveStr='.pdf'):
	'''
	Creates a dispersion plot (or bandgap plot) for the Dispersion Expression. Plot can be optionally saved
	INPUTS:
		alpha: 	float, value for the thick-vertex constant alpha
		wPts: 	int (default 1000), number of points to plot DR value at in omega
		kPts: 	int (default 1000), number of points to plot DR value at in wavenumber
		saveFig: 	bool (default False), if True the plot that is produced will be saved
		saveStr: 	str (default '.pdf'), if saveFig is True then the file produced will be saved with this name
	OUTPUTS:
		fig: 	matplotlib.pyplot.figure object, belonging to the plot that was produced
		ax: 	matplotlib.pyplot.axes object, belonging to the axes on which the DR was plotted
	'''
	
	wRange = np.linspace(0,6*pi,wPts)
	kRange = np.linspace(0,3*pi,kPts)
	
	dRIndicator = np.zeros((wPts, kPts), dtype=int)
	dRSurf = np.zeros((wPts, kPts), dtype=float)
	
	#compute values of the Dispersion Relation across the w,k surface we have
	for k in range(kPts):
		kVal = kRange[k]
		firstInd = np.nonzero(wRange>kVal)[0][0] #first index of wRange that has w>kVal
		dRSurf[firstInd:, k] = DispExpr(wRange[firstInd:], kVal, alpha) #DispExpr values that we can compute
		dRSurf[:firstInd, k] = np.NaN #values that the DR cannot be evaluated at
		dRIndicator[:firstInd, k] = -1 #set these to be -1 for consistency in plotting the bandgaps
	
	#now indicate the points where eigenvalues exist (+1) or don't exist (-1). NaNs might play up but numpy handles them correctly in array comparison slicing
	dRIndicator[abs(dRSurf)<=1] = 1
	dRIndicator[abs(dRSurf)>1] = -1
	
	#dRIndicator is now ready to be contour or surface plotted via Plot3D if we want
	#fig, ax = Plot3D(wRange, kRange, dRIndicator)
	
	#plot as 2D heatmap via seaborn
	#indicatorFrame = pd.DataFrame(np.transpose(dRIndicator), columns=kRange/pi, index=wRange/pi)
	#surfFrame = pd.DataFrame(np.transpose(dRSurf), columns=kRange/pi, index=wRange/pi)
	#fig, ax = PlotHeatmap(indicatorFrame)
	
	#plot as contour using matplotlib
	fig, ax = PlotContour(wRange/pi, kRange/pi, np.transpose(dRIndicator))

	#labels that apply regardless of which plot we made
	ax.set_xlabel(r'Frequency, $\frac{\omega}{\pi}$')
	ax.set_ylabel(r'Wavenumber, $\frac{\kappa}{\pi}$')
	ax.set_title(r'Bandgap Plot')
	
	#if we want to save things
	try:
		if saveFig==True and len(saveStr)>4:
			fig.savefig(saveStr)
		elif saveFig:
			raise ValueError('Invalid filename, try something with an extension and at least 2 characters.')
	except ValueError:
		warn('Invalid filename provided, figure not saved!')
	except:
		warn('Unexpected error occurred whilst saving, run in debug mode.')
		
	#return plot if sucessful
	return fig, ax

def PlotHeatmap(df):
	'''
	Plots the data in the dataframe provided and returns the plot handles.
	INPUTS:
		df: 	pandas dataframe in xyz format
	OUTPUTS:
		fig: 	matplotlib.pyplot.figure object, belonging to the plot that was produced
		ax: 	matplotlib.pyplot.axes object, belonging to the axes on which the DR was plotted
	'''
	
	fig, ax = plt.subplots(1)
	FigWindowSettings(ax)
	ax = sns.heatmap(df, xticklabels=False, yticklabels=False, cbar=False) #needs labels redrawn, etc
	ax.invert_yaxis()
	
	return fig, ax

def PlotContour(x, y, z):
	'''
	Creates a (filled) contour plot of the surface z
	INPUTS:
		x: 	(nx,) float numpy array, range of w values that surfData is plotted over
		y: 	(ny,) float numpy array, range of k values that surfData is plotted over
		z: 	(ny, nx) float numpy array, surface values to be plotted
	OUTPUTS:
		fig: 	matplotlib.pyplot.figure object, belonging to the plot that was produced
		ax: 	matplotlib.pyplot.axes object, belonging to the axes on which the DR was plotted
	'''
	
	fig, ax = plt.subplots(1)
	FigWindowSettings(ax)
	ax.contourf(x, y, z, levels=1, colors=['blue','yellow'])
	
	return fig, ax

def Plot3D(wRange, kRange, surfData):
	'''
	Plots a 3D surface (viewed from above) using matplotlib.pyplot.
	INPUTS:
		wRange: 	(wPts,) float numpy array, range of w values that surfData is plotted over
		kRange: 	(kPts,) float numpy array, range of k values that surfData is plotted over
		surfData: 	(wPts, kPts) float numpy array, surface values to be plotted
	OUTPUTS:
		fig: 	matplotlib.pyplot.figure object, belonging to the plot that was produced
		ax: 	matplotlib.pyplot.axes object, belonging to the axes on which the DR was plotted	
	'''

	W, K = np.meshgrid(wRange, kRange)
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot_surface(W, K, surfData, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	ax.view_init(elev=-90, azim=0)
	fig.show()
	
	return fig, ax

#stuff we actually want to do when we run the script
if __name__=='__main__':
	#default use latex in axis labels
	rc('text', usetex=True) 
	
	#some parameter values to set
	wavenumber = pi
	alpha = -4
	wPts = 1000
	kPts = 1000
	saveStr = input('Save figure as (.pdf appended automatically): ') + '.pdf'
	print('Filename provided: ', saveStr)
	
	fig, ax = FixedWavenumberSlice(wavenumber, alpha, wPts, saveFig=True, saveStr=saveStr)
	#fig, ax = DispersionPlot(alpha, wPts=wPts, kPts=kPts, saveFig=True, saveStr=saveStr)