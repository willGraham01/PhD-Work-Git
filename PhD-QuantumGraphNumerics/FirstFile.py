#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 13:47:48 2019

@author: will

First stab at some numerics for the scalar case of a quantum graph. Wish me luck.
"""

#first let's have a think about what we're going to need to give this program

#we'll need the adjacency matrix. Also the positions of the vertices (from which we can compute the lengths of the edges). Any coupling constants will also need to be stored, as well as the effect of the quasi-momentum on each edge I suppose.

#for the time being, let's build a lookup program. As in, we'll assume we have an analogue of Thm 3.2 in EKK paper and we can just perform a lookup to assemble the function. In general, we'll need to do an ODE solve on each edge with various boundary conditions.


#first a function that can draw the graph that we are considering!

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def DrawGraph(adjMat, vPos, numberingOffset=0):
	'''
	Draws the graph given by the adjacency matrix and the vertex positions provided.
	Assumes no loops in the graph - IE the adjacency matrix has entries consisting of 0's and 1's.
	INPUTS:
		adjMat 	: (n,n) numpy array - adjacency matrix for the graph that is to be drawn
		vPos 	: (n,2) numpy array - vertex co-ordinates as (x,y) pairs
	OPTIONAL INPUTS:
		numberingOffset 	: int - vertices will be labelled by their index in vPos plus this value. That is, vertex numbering in the resulting diagram starts from this value.
	OUTPUTS:
	'''
	
	nVerts = np.shape(vPos)[0] 	#get the number of vertices of this graph
	ax = plt.gca()				#create plot axes
	
	for i in range(0,nVerts):
		#for each vertex in the graph
		for j in range(i+1,nVerts):
			#we only need to consider either the upper or lower triangular part of the adjacency matrix is we assume there are no loops
			if adjMat[i,j]!=0:
				#these vertices are connected by an edge. Join them in the diagram
				edge = mlines.Line2D(vPos[[i,j],0],vPos[[i,j],1],color='r')
				ax.add_line(edge)
				#plt.plot(vPos[[i,j],0],vPos[[i,j],1],'r-')
		#after having drawn the edges that connect to this vertex, draw the vertex itself with it's label.
		ax.annotate('v'+str(numberingOffset + i), (vPos[i,0],vPos[i,1]), horizontalalignment='left', verticalalignment='bottom' )
		
	ax.scatter(vPos[:,0],vPos[:,1],s=20,c='k')	#scatter plot the graph vertices
	
	plt.show()
	
	return ax