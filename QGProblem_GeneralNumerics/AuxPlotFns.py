#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 11:42:54 2020

@author: will

This file contains all auxillary plotting functions that I want to use for the numerical results that are produced throughout my PhD.
These generally include functions take in numpy arrays of x, y, (and ,z) data and return fig, ax handles in matplotlib displaying that data, to speed up the process and save writing this out each time.
There are also some layout functions to standardis the look of my plotting so I don't have any regrets at any time soon.

NOTE: Importing this package by default sets matplotlib to render TeX in axis labels
Rough table of contents:
	- FigWindowSettings
	- GenTitStr
	- PlotContour
	- PlotSurf
	- PlotHeatmap
	- FastFunctionPlot
"""

#numpy is always needed
import numpy as np
from numpy import pi

#matplotlib for plotting in 3d and 2d
import matplotlib.pyplot as plt
from matplotlib import rc, cm

import seaborn as sns

#Sets the default window style for 2D plots.
# No right or top boarder on plots, tick marks on remaining boarders
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

def PlotSurf(x, y, z):
	'''
	Plots a 3D surface (viewed from above) using matplotlib.pyplot.
	INPUTS:
		x: 	(nx,) float numpy array, range of w values that surfData is plotted over
		y: 	(ny,) float numpy array, range of k values that surfData is plotted over
		z: 	(ny, nx) float numpy array, surface values to be plotted
	OUTPUTS:
		fig: 	matplotlib.pyplot.figure object, belonging to the plot that was produced
		ax: 	matplotlib.pyplot.axes object, belonging to the axes on which the DR was plotted	
	'''

	X, Y = np.meshgrid(x, y)
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot_surface(X, Y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	ax.view_init(elev=-90, azim=0)
	
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

def FastFunctionPlot(x, f):
	'''
	For a (vectorised) function f, produce a plot of f over the range supplied in x
	INPUTS:
		x: 	(n,) float numpy array, points to evaluate the function f at
		f: 	lambda function, vectorised function handle to be evaluated
	OUTPUTS:
		fig: 	matplotlib.pyplot.figure object, belonging to the plot that was produced
		ax: 	matplotlib.pyplot.axes object, belonging to the axes on which the DR was plotted	
	'''

	fig, ax = plt.subplots(1)
	FigWindowSettings(ax)
	ax.plot(x, f(x))	
	
	return fig, ax

#always want to use LATEX in figure captions, labels, etc
if __name__=='__main__':
	rc('text', usetex=True) 