#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 12:34:05 2019

@author: will

This file will serve to create the datastructures needed for the quantum graph numerics.
In particular, it will setup the vertex class and provide methods for assembling a graph consisting of vertices.

Presently we are still assuming we have -d^2/dx^2 on each vertex still (subject to quasimomentum shifts, of course).
"""

import csv #reading in .csv files
from warnings import warn
import numpy as np
from numpy.linalg import norm
from numpy import sin, tan
import matplotlib.pyplot as plt
import matplotlib.patches as pts #for arrows
from matplotlib import rc #for text options in plots
from scipy.optimize import fsolve #for nonlinear inverse iteration
		
class Graph:
	'''
	An instance of the Graph class consists of the positions of the vertices in the graph, their connections to each other, and information on the lengths of edges between vertices and (if applicable) the value of the quasimomentum parameter along each edge.
	For more information, one can see the function GraphConstructorHelp().
	Attributes:
		nVert 	: int, number of vertices in the graph
		vPos 	: (2,n) numpy array, each column is the (x,y) position of one of the vertices in the graph. By default, vertices will have ID's from 0 to nVert-1.
		adjMat 	: (nVert,nVert) numpy array, the "modified" adjacency matrix for the graph that also encodes directions
		lenMat 	: dict of numpy arrays (x,), stores the lengths of connected vertices - lenMat[i,j] points to a numpy array of length x where x is the number of edges connecting i to j, each element of the array being the length of an edge
		theMat 	: dict of numpy arrays (2,x), stores the coefficents of the two components of the quasimomentum theta for each edge. theMat[i,j] points to a numpy array of shape (2,x) where each column is a pair of coefficients for an edge, and x is the number of edges connecting i to j.
		coupConsts 	: (nVert, nVert)-valued function of (w), evaluates the (diagonal) matrix of coupling constants at the given value of w.
	Methods:
	'''
	#Default attribute values for a graph - although these should ALWAYS be overwritten in construction.
	nVert = 1
	vPos = np.asarray([])
	adjMat = np.zeros((nVert,nVert))
	lenMat = np.zeros((nVert,nVert))
	theMat = np.zeros((nVert,nVert,2))
	coupConsts = np.zeros((nVert,nVert))
	
	def __init__(self, fileName):
		'''
		Construction method for the Graph class. An instance of the Graph class will be created with its attributes generated to be consistent with the inputs.
		INPUTS:
			fileName 	: file or path to file that contains the graph information as dictated by the documentation. Graph will be assembled from this file. See GraphConstructorHelp() for more information.
		'''
		#read the input file line-by-line
		lines = []
		with open(fileName, "r") as f:
			reader = csv.reader(f, delimiter=",")
			for line in reader:
				lines.append(line)
		#we have now read in the input .csv file
		#the first element of lines is just ['Vertices','','','',''] so we can bypass this I guess. NOTE that everything is imported as a string, so need to convert the stored "numbers" to actual floats too!
		linesInd = 1;	nLines = len(lines)
		nVert = 0; 	vertEnd = False
		xList = [];	yList = []
		
		#create lists of x and y o-ordinates, and thus the vPos variable
		while (linesInd<nLines) and (not vertEnd):
			#whilst we still haven't reached the end of the vertices segment of the file
			if lines[linesInd][0]!="Edges":
				#this line defines a new vertex
				xList.append(float(lines[linesInd][1]))
				yList.append(float(lines[linesInd][2]))
				nVert += 1
			else:
				#this line signals the start of the edges segment, consolodate the information we have
				vertEnd = True
				self.vPos = np.asarray([xList, yList])
				self.nVert = nVert
			linesInd += 1
		print('Found %d vertices in this graph file' % (self.nVert))
		#we have now created all the vertices and their positions, now it's time to form adjMat, lenMat and theMat
		
		#at this point, linesInd is at the first line which defines an edge
		edgeStart = linesInd
		edgeEnd = 0
		stillEdges = True
		while (linesInd<nLines) and stillEdges:
			#quickly find the index of the line where edge definitions stop and coupling constants start
			if lines[linesInd][0]=="Coupling":
				#this is where edges end, save these
				edgeEnd = np.copy(linesInd)
				stillEdges = False
			linesInd += 1
		if stillEdges:
			#got to end of file without Coupling keyword - set edgeEnd to end of file.
			edgeEnd = nLines
			print('No coupling constants provided in input')
		#when we get to here, we know that edge definitions start on line edgeStart and end on line edgeEnd-1
		#meanwhile, linesInd is the first line referring to coupling constants, or is after the end of the file.

		#fill in edge information 
		aMat = np.zeros((self.nVert, self.nVert), dtype=int)
		for eInd in range(edgeStart, edgeEnd):
			#for all the lines that define edges
			vLeft = int(lines[eInd][0]); vRight = int(lines[eInd][1])
			if vLeft!=vRight:
				aMat[vLeft,vRight] += 1	
			else:
				#must have vRight==vLeft - give waring because loops :O
				warn('Loop encoded at vertex %d but unsupported' % (vLeft))
		#aMat is good to go as adjMat
		self.adjMat = np.copy(aMat)
		#lMat needs to have each [j][k] list converted to a numpy array of length aMat[j,k]
		#theMat needs to be constructed as a list of lists, [j][k] being a (2,adjMat[j,k]) numpy array whose columns are the quasimomentum coefficients for each edge.
		self.lenMat = dict()
		self.theMat = dict()
		for j in range(self.nVert):
			for k in range(self.nVert):
				if self.adjMat[j,k]!=0:
					#if there is an edge here, then we need to do something
					#initialise dictionaries
					self.lenMat[j,k] = np.zeros((self.adjMat[j,k],), dtype=float)
					self.theMat[j,k] = np.zeros((2, self.adjMat[j,k]), dtype=float)
					#fill in dictionary information - this process has a more efficient method of execution but oh well for now
					insertInd = 0
					for eInd in range(edgeStart, edgeEnd):
						vLeft = int(lines[eInd][0]); vRight = int(lines[eInd][1])
						if (j==vLeft) and (k==vRight):
							#this line has information on an edge joining these two vertices in the direction we expect
							
							if float(lines[eInd][2])==0:
								#field left blank implies use Euclidean distance
								self.lenMat[j,k][insertInd] = norm(self.vPos[:,j] - self.vPos[:,k])
							else:
								#take value given for this edge
								self.lenMat[j,k][insertInd] = float(lines[eInd][2])
							self.theMat[j,k][0,insertInd] = float(lines[eInd][3])
							self.theMat[j,k][1,insertInd] = float(lines[eInd][4])
							insertInd += 1

		#if we have coupling constants, we should sort them out too!
		if linesInd>=nLines:
			#if we reached the end of the file in the previous step, there is no coupling constant information given. Return the constant (0) function
			def CoupMatrix(w):
				'''
				This graph has no coupling constants, so the matrix of coupling constants is zero everywhere.
				'''
				return np.zeros((self.nVert, self.nVert), dtype=complex)
			self.coupConsts = CoupMatrix
		else:
			#we didn't reach the end of the file - there are some coupling constants to append.
			def CoupMatrix(w):
				'''
				Evaluates the (diagonal) matrix of coupling constants for this graph.
				'''
				matOut = np.zeros((self.nVert, self.nVert), dtype=complex)
				for eInd in range(linesInd, nLines):
					vID = int(lines[eInd][0])
					matOut[vID,vID] = PolyEval(w, lines[eInd][1:])
				return matOut
			self.coupConsts = CoupMatrix

		return #construction method ends here
		
	def __repr__(self):
		'''
		Default output of a call to a graph object.
		'''
		return "<Graph instance with %d vertices and %d edges>" % (self.nVert, np.sum(np.sum(self.adjMat)))
		
	def __str__(self):
		'''
		Print text that is displayed for Graph objects when called with the print() function. Currently produces an image of the graph and returns some informative text
		'''
		self.Draw(show=False)
		
		return "Graph instance with %d vertices and %d edges" % (self.nVert, np.sum(np.sum(self.adjMat)))
	
	def Connections(self, vID):
		'''
		Given a graph and the ID of one of the vertices, return two lists containing the indices of those vertices that are connected to vertex vID, the first the connections with vID the left endpoint and the second the connections with vID the right endpoint
		INPUTS:
			vID 	: int, ID of the vertex whose connections we want to find
		OUTPUTS:
			lCon 	: int list, IDs of the vertices which have connections to vID, with vID the left endpoint of the edge
			rCon 	: int list, IDs of the vertices which have connections to vID, with vID the right endpoint of the edge
		'''
		
		#we can find the connections that a vertex has by looking at the modified "adjacency matrix"
		lCon = [];	rCon = [];
		
		#work along row and column vID to determine the connections
		for i in range(0,vID):
			#first we work from the left (row-wise) and top (column-wise) until we reach the diagonal
			if self.adjMat[i,vID]>0:
				#entry [i,vID] indicates a connection in the upper triangle of the matrix - this is a connection with i the left endpoint and vID the right endpoint
				rCon.append(i)
			if self.adjMat[vID,i]>0:
				#entry [vID,i] indicates a connection in the lower triangle of the matrix - this is a connection with i the right endpoint and vID the left endpoint
				lCon.append(i)
		for i in range(vID+1,self.nVert):
			#now go past the diagonal (no loops assumed, and in case there are dodgy diagonal entries) and keep searching
			if self.adjMat[i,vID]>0:
				#entry [i,vID] indicates a connection in the lower triangle of the matrix - this is a connection with i the right endpoint and vID the left endpoint
				lCon.append(i)
			if self.adjMat[vID,i]>0:
				#entry [vID,i] indicates a connection in the upper triangle of the matrix - this is a connection with i the left endpoint and vID the right endpoint
				rCon.append(i)
		return lCon, rCon
	
	def ConstructM(self, dervToo=False):
		'''
		Constructs the Weyl-Titchmarsh M-function (matrix in this case) for the graph.
		INPUTS:
			dervToo 	: (optional) bool, if true then we will also construct and return the function M'(w). Default False
		OUTPUTS:
			M 	: lambda function, M(w,theta) is the value of the M-matrix function at the value w and for the quasimomentum theta provided
			Mprime 	: (optional) lambda function, Mprime(w,theta) is the value of the function M'(w, theta) - only returned if dervToo is True.
		'''
		#first, we need to come up with a plan for efficiently constructing this function
		#see some paper... good luck trying to explain WTF is going on to your future self
		
		#THE PLAN: We need to construct M effectively so that we don't have to pass through many many comparison checks each time we want to evaluate the function. The way we do this will be to construct a list of lists.
		a_list = []
		for i in range(self.nVert):
			a_list.append([])
		masterList = []
		for i in range(self.nVert):
			masterList.append(list(a_list))
		#masterList is a list of length nVert, where each entry is another list of length nVert. As such, masterList has the structure of an nVert-by-nVert matrix.
		#The plan will now be to store the information we need for the sums involved in each element of M in the corresponding member of masterList
		nonZeroElements = []
		#nonZeroElements will be a list of (i,j) indices corresponding to the non-identically zero entries of the M-matrix. The idea being we don't need to waste computation time
		
		for j in range(self.nVert):
			for k in range(self.nVert):
				#first we check if j and k are connected - we need to check connections both ways of course
				if j==k:
					#this entry will be non-zero
					nonZeroElements.append([j,k])
					#this is a diagonal entry, so we need to store the lengths of all the edges that connect to vertex k
					
					#get all of the connections that vertex k=j has
					lCon, rCon = self.Connections(k)
					#create a vector containing all the lengths, and save it to the list location
					lens = []
					#append all edge lengths with v_k the left edge
					for vID in lCon:
						for i in range(np.shape(self.lenMat[k,vID])[0]):
							lens.append(self.lenMat[k,vID][i])
					#append all edge lengths with v_k the right edge
					for vID in rCon:
						for i in range(np.shape(self.lenMat[vID,k])[0]):
							lens.append(self.lenMat[vID,k][i])
					#now we just save lens to the relevant place in masterList
					masterList[k][k] = np.copy(lens)
					
					#in the diagonal entries we are storing a vector of the edge lengths that connect to vertex j=k. This is all we need to compute this entry of M, because without loops the quasimomentum parameter doesn't enter into this value of the matrix.
					
				elif (self.adjMat[j,k] + self.adjMat[k,j])>0:
					#these are connected but j and k are different vertices
					nonZeroElements.append([j,k])
					#we now need to form two lists to account for the direction of edges into and out of the vertices, plus a further 4 for the quasimomentum that may change on each of the vertices.
					
					kLeft = []; kRight = [] #connection stores
					t1Left = []; t2Left = [] #quasimomentum coefficients for left
					t1Right = []; t2Right = [] #quasimomentum coefficients for right
					
					if self.adjMat[j,k]>0:
						#there is a connection here
						if j<k:
							#connection is in the upper triangle, so j is the left vertex and k is the right vertex
							kRight = self.lenMat[j,k]
							t1Right = self.theMat[j,k][0,:]
							t2Right = self.theMat[j,k][1,:]
							if self.adjMat[k,j]>0:
								#there is also a reverse connection, so we need to append to the left lists too
								kLeft = self.lenMat[k,j]
								t1Left = self.theMat[k,j][0,:]
								t2Left = self.theMat[k,j][1,:]
						else:
							#connection is in the lower triangle, so j is the right vertex and k is the left vertex
							kLeft = self.lenMat[j,k]
							t1Left = self.theMat[j,k][0,:]
							t2Left = self.theMat[j,k][1,:]
							if self.adjMat[k,j]>0:
								#there is also a reverse connection, so we need to append to the right lists too
								kRight = self.lenMat[k,j]
								t1Right = self.theMat[k,j][0,:]
								t2Right = self.theMat[k,j][1,:]
					else: 
						# self.adjMat[k,j]>0 (as the previous sum is positive, but adjMat[j,k]=0 as entries are >=0), but there is no reverse connection as it would have been caught in the previous case
						if k<j:
							#connection is in the upper triangle, so k is the left vertex and j is the right vertex
							kLeft = self.lenMat[k,j]
							t1Left = self.theMat[k,j][0,:]
							t2Left = self.theMat[k,j][1,:]
						else:
							#connection is in the lower triangle, so k is the right vertex and j is the left vertex
							kRight = self.lenMat[k,j]
							t1Right = self.theMat[k,j][0,:]
							t2Right = self.theMat[k,j][1,:]
					#although any of these we have already assigned have the correct format, those that are not assigned need to be put through this, so it doesn't hurt to do it to all of them.
					kLeft = np.asarray(kLeft)
					kRight = np.asarray(kRight)
					t1Left = np.asarray(t1Left)
					t2Left = np.asarray(t2Left)
					t1Right = np.asarray(t1Right)
					t2Right = np.asarray(t2Right)
					masterList[j][k] = [kLeft, t1Left, t2Left, kRight, t1Right, t2Right]
					
					#in the off-diagonal terms, masterList stores a further list of 6 entries; effectively in two groups. The first 3 are numpy arrays of the lengths of edges with vertex k on the left, followed by the QM parameters in for the first and second components in the next two. The final 3 are similar, but for edges with k on the right.
		
		#having done this for loop, masterList is setup. It should now be a case of defining a function and returning it as a lambda function...
		def EvalMatrix(w,theta=np.zeros((2), dtype=float)):
			'''
			Evaluates the M-matrix at the value w, given the quasimomentum theta
			INPUTS:
				w 	: float, value of w to evaluate M-matrix at
				theta 	: (optional) (2,) numpy array, value of the quasimomentum vector theta. Default [0,0]
			OUTPUTS:
				mat 	: M-matrix at (w,theta)
			'''
			mat = np.zeros((self.nVert,self.nVert), dtype=complex) #in general this matrix will be Hermitian
			for j,k in nonZeroElements:
				#fill in non-identically zero elements
				if j==k:
					#diagonal entry
					# = -w * sum_{k~l} cot(w*l_{kl})
					mat[j,k] = -w * np.sum( cot(w*masterList[k][k]) )
				else:
					#off-diagonal entry
					# = w * sum_{k~j, k left}e^{-i*theta_kj*l_kj}cosec(w*l_kj) + w * sum_{k~j, k right}e^{i*theta_kj*l_kj}cosec(w*l_kj)
					mat[j,k] = w * np.sum( np.exp( -(1.j*theta[0])*masterList[j][k][0]*masterList[j][k][1] ) * np.exp( -(1.j*theta[1])*masterList[j][k][0]*masterList[j][k][2] ) * cosec(w*masterList[j][k][0]) ) #-ve  in exp for left
					mat[j,k] += w * np.sum( np.exp( (1.j*theta[0])*masterList[j][k][3]*masterList[j][k][4] ) * np.exp( (1.j*theta[1])*masterList[j][k][3]*masterList[j][k][5] ) * cosec(w*masterList[j][k][3]) ) #+ve in exp for right
			return mat

		#having created the lambda function, return it as the output
		M = EvalMatrix
		
		if dervToo:
			#if this value is true, we also want to construct the derivative M'(w, theta) wrt w. Thankfully, we have all the information we need from above, so this shouldn't be too hard...
			def DervEvalMatrix(w,theta=np.zeros((2), dtype=float)):
				'''
				Evaluates the derivative (wrt w) of the M-matrix, given the quasimomentum theta
				INPUTS:
					w 	: float, value of w to evaluate M-matrix derivative at
					theta 	: (optional) (2,) numpy array, value of the quasimomentum theta. Default [0,0]
				OUTPUTS:
					mat 	: derivative of M at (w,theta), M'(w,theta)
				'''
				mat = np.zeros((self.nVert,self.nVert), dtype=complex) #in general this matrix will be Hermitian
				for j,k in nonZeroElements:
					#fill in non-identically zero elements
					if j==k:
						#diagonal entry
						# = -sum_{k~l} cot(w*l_kl) + w* sum_{k~l} l_kl cosec^2(w*l_kl)
						mat[j,k] = -1.0 * np.sum( cot(w*masterList[k][k]) )
						mat[j,k] += w * np.sum( masterList[k][k] * np.power(cosec(w*masterList[k][k]),2) )
					else:
						#off-diagonal entry
						# = sum_{k~j, k left}e^{-i*theta_kj*l_kj}cosec(w*l_kj) + sum_{k~j, k right}e^{i*theta_kj*l_kj}cosec(w*l_kj)  - w*sum_{k~j, k left}e^{-i*theta_kj*l_kj}l_kj*cosec(w*l_kj)*cot(w*l_kj) - w*sum_{k~j, k right}e^{i*theta_kj*l_kj}l_kj*cosec(w*l_kj)*cot(w*l_kj)
						mat[j,k] = np.sum( np.exp( -(1.j*theta[0])*masterList[j][k][0]*masterList[j][k][1] ) * np.exp( -(1.j*theta[1])*masterList[j][k][0]*masterList[j][k][2] ) * cosec(w*masterList[j][k][0]) ) #-ve  in exp for left
						mat[j,k] += np.sum( np.exp( (1.j*theta[0])*masterList[j][k][3]*masterList[j][k][4] ) * np.exp( (1.j*theta[1])*masterList[j][k][3]*masterList[j][k][5] ) * cosec(w*masterList[j][k][3]) ) #+ve in exp for right
						mat[j,k] -= w * np.sum( np.exp( -(1.j*theta[0])*masterList[j][k][0]*masterList[j][k][1] ) * np.exp( -(1.j*theta[1])*masterList[j][k][0]*masterList[j][k][2] ) * masterList[j][k][0] * cosec(w*masterList[j][k][0]) * cot(w*masterList[j][k][0]) )
						mat[j,k] -= w * np.sum( np.exp( -(1.j*theta[0])*masterList[j][k][3]*masterList[j][k][4] ) * np.exp( -(1.j*theta[1])*masterList[j][k][3]*masterList[j][k][5] ) * masterList[j][k][3] * cosec(w*masterList[j][k][3]) * cot(w*masterList[j][k][3]) )
				return mat				
			
			Mprime = DervEvalMatrix
			return M, Mprime
		
		#if we get to here we never built Mprime, so just return M
		return M
	
	def Draw(self, offSet=1, tex=True, show=False, radFac=0.25):
		'''
		Draws the structure of the graph as a matplotlib figure.
		INPUTS:
			offSet 	: (optional) int, number to add to each of the vertex IDs for consistency between program and analytic problem. Default 1
			tex 	: (optional) bool, if True then the diagram will be plotted using LaTeX rendering, otherwise use the matplotlib defaults. Default True
			show 	: (optional) bool, if True then the diagram is displayed by the method before returning the outputs - use for suppression of plots in loops. Default False
			radFac 	: (optional) float, determines the "curviness" of the arrows in the displayed diagram. Smaller values result in straighter arrows and are useful for complex or cluttered graphs. Default 0.25
		OUTPUTS:
			fig 	: matplotlib figure, the assembled diagram of the graph.
		'''
		#setup axes
		xMin = min(np.min(self.vPos[0,:]),0.0); xMax = max(np.max(self.vPos[0,:]),1.0)
		yMin = min(np.min(self.vPos[1,:]),0.0); yMax = max(np.max(self.vPos[1,:]),1.0)
		#create axis instance
		fig = plt.figure()
		ax = plt.gca()
		ax.set_xlim(xMin-0.1,xMax+0.1)
		ax.set_ylim(yMin-0.1,yMax+0.1)
		ax.set_aspect(1)
		if tex:
			#tell python to write things out in LaTeX
			rc('text', usetex=True)
			rc('font', family='serif')
			ax.xaxis.set_label_text('$x$')
			ax.yaxis.set_label_text('$y$')
		else:
			rc('text', usetex=False)
			ax.xaxis.set_label_text('x')
			ax.yaxis.set_label_text('y')
		ax.title.set_text('Diagram of graph')
		#define arrow style for connected edges
		style="Simple,tail_width=0.5,head_width=4,head_length=8"
		kw = dict(arrowstyle=style, color="r", zorder=1)
		
		#add the vertices to the plot
		for i in range(self.nVert):
			#for each vertex in the graph, draw the vertex with it's label.
			if tex:
				ax.annotate('$v_'+str(offSet+i)+'$', (self.vPos[0,i],self.vPos[1,i]), horizontalalignment='left', verticalalignment='bottom' )
			else:
				ax.annotate('v_'+str(offSet+i), (self.vPos[0,i],self.vPos[1,i]), horizontalalignment='left', verticalalignment='bottom' )
		#scatter plot the graph vertices to put them on the graph
		ax.scatter(self.vPos[:,0],self.vPos[:,1],s=20,c='k',zorder=2)

		#store all the edges (with directions) that we need to draw
		arrowList = []
		#for changing the arcs of the arrows when there are multiple connections
		conStyle = "arc3,rad="
		#now let's make a list of all the arrows we want to draw :)
		for i in range(self.nVert):
			vi = (self.vPos[0,i],self.vPos[1,i])
			#save time looping by not revisiting vertices we've already drawn
			for j in range(i+1,self.nVert):
				vj = (self.vPos[0,j],self.vPos[1,j])
				#note that j>=i+1 here so we can use that to determine which way the arrows should point
				if self.adjMat[i,j]>0 and self.adjMat[j,i]>0:
					#2-way connection, draw multiple arrows
					for k in range(1,self.adjMat[i,j]+1):
						#i<=j-1 so i<j and thus we are in the upper triangle. Hence vi is left and vj is right. Draw as many arrows as we need
						kConStyle = conStyle + str(k*radFac)
						arrow_ijk = pts.FancyArrowPatch(vi, vj, connectionstyle=kConStyle,**kw)
						arrowList.append(arrow_ijk)
					for k in range(1,self.adjMat[j,i]+1):
						kConStyle = conStyle + str(k*radFac)
						arrow_jik = pts.FancyArrowPatch(vj, vi, connectionstyle=kConStyle,**kw)
						arrowList.append(arrow_jik)
				elif self.adjMat[i,j]>0:
					#only one direction. As j>=i+1, we are in the upper triangle so vi is left and vj is right. Make sure the arrow points this way, but they might still need to be curved
					for k in range(self.adjMat[i,j]):
						kConStyle = conStyle + str(k*radFac)
						arrow_ijk = pts.FancyArrowPatch(vi, vj, connectionstyle=kConStyle,**kw)
						arrowList.append(arrow_ijk)
				elif self.adjMat[j,i]>0:
					#only one direction. As j>i+1, we are in the lower triangle so vj is left and vi is right. Make sure the arrow points this way, but they might still need to be curved
					for k in range(self.adjMat[j,i]):
						kConStyle = conStyle + str(k*radFac)
						arrow_jik = pts.FancyArrowPatch(vj, vi, connectionstyle=kConStyle,**kw)
						arrowList.append(arrow_jik)
				#in the else case we simply do nothing!
		#add the arrows to the plot
		for arrow in arrowList:
			ax.add_patch(arrow)
		if show:
			#show the resulting plot
			fig.show()
		return fig
	
#class definition for Graph ends here
		
#functions that have a computational purpose
def cot(x):
	'''
	Cotangent function, element-wise
	INPUTS:
		x 	: numpy array, values to input into cotangent (radians)
	OUPUTS:
		cotx 	: numpy array, same shape as x with values cot(x) element wise
	'''	
	return	1/tan(x)#np.cos(x)/np.sin(x)

def cosec(x):
	'''
	Cosecant function, element-wise
	INPUTS:
		x 	: numpy array, values in radians
	OUTPUTS:
		cscx 	: numpy array, same shape as x with values cosec(x) element wise
	'''
	return 1/sin(x)

##Solver Methods and associated subfunctions
def CompToReal(x):
	'''
	Given a (n,) complex numpy array, return a (2n,) numpy array of floats containing the real and imaginary parts of the input complex vector.
	INPUTS:
		x 	: (n,) complex numpy array, array to be converted to (2n,) real array
	OUTPUTS:
		z 	: (2n,) numpy array, containing real and imaginary parts of the values in x, according to real(x[k]) = z[2k], imag(x[k]) = z[2k+1]
	'''
	
	n = np.shape(x)[0]
	z = np.zeros((2*n,), dtype=float)
	for k in range(n):
		z[2*k] = np.real(x[k])
		z[2*k+1] = np.imag(x[k])
	
	return z

def RealToComp(z):
	'''
	Given a (2n,) numpy array, return a (n,) complex numpy array of numbers whose real and imaginary parts correspond index-neighbour pairs of the input array.
	INPUTS:
		z 	: (2n,) numpy array, array to be converted to (n,) complex array
	OUTPUTS:
		x 	: (n,) complex numpy array, containing complex numbers formed from the entries of z, according to x[k] = z[2k]+i*z[2k+1]
	'''

	if np.shape(z)[0] % 2 !=0:
		raise ValueError('Cannot convert array of non-even length (%.1f) to complex array' % np.shape(z)[0])
	else:
		n = int(np.shape(z)[0]/2) #safe to use int() after the previous check
		x = np.zeros((n,), dtype=complex)
		for k in range(n):
			x[k] = z[2*k] + 1.j*z[2*k+1]

	return x

def UnitVector(i,n=3):
	'''
	Creates the cannonical (i+1)-th unit vector in R^n, albeit as an array of complex data types
	INPUTS:
		i 	: int, python index corresponding to the (i+1)th component of the unit vector t be set to 1 - all others are 0
		n 	: (optional) int, size of the unit vector or dimension of R^n. Default 3
	OUTPUTS:
		e 	: i-th cannonical unit vector in R^n as an array of complex data types
	'''
	e = np.zeros((n,), dtype=complex)
	e[i] = 1.+0.j
	
	return e

def PolyEval(x, cList):
	'''
	Given a value x and a list of coefficients, return the value of the polynomial expression cList[0]+x*cList[1]+x^2*cList[2]+...
	INPUTS:
		x 	: complex float, value to evaluate polynomial map at
		cList 	: list (to be cast to floats), polynomial coefficients
	OUTPUTS:
		pVal 	: polynomial evaluation at x
	'''
	pVal = 0.
	for i, coeff in enumerate(cList):
		if len(coeff)>0:
			pVal += float(coeff) * (x**i)
	
	return pVal

#nonlinear inverse interation solver...
def NLII(M, Mprime, v0, u, w0=np.pi, theta=np.zeros((2), dtype=float), maxItt=100, tol=1.0e-8, conLog=True, talkToMe=False):
	'''
	Solve the nonlinear eigenvalue problem M(w,theta)v = 0 for an eigenpair (w,v) using the Nonlinear Inverse Iteration method (see Guttel & Tisseur, 2017).
	INPUTS:
		M 	: lambda function, evaluates M(w,theta) at arguments (w,theta)
		Mprime 	: lambda function, evaluates d/dw M(w,theta) at arguments (w,theta)
		v0 	: (n,) numpy array, initial guess for the eigenvector
		u 	: (n,) numpy array, the vector u is used to normalise the output eigenvector and can be used to avoid repeat convergence to the same eigenpair in the case of holomorphic M
		w0 	: (optional) float, starting guess for the eigenvalue w. Default np.pi
		theta 	: (optional) (2,) numpy array, the quasimomentum value for this solve. Default [0,0]
		maxItt 	: (optional) int, maximum number of iterations to perform. Default 100
		tol 	: (optional) float, solution tolerance. Default 1.0e-8
		conLog 	: (optional) bool, if True then a list storing the information after each iteration plus any errors or warnings will be returned. Default True.
		talkToMe 	: (optional) bool, if True then the method will printout information about the converged solution at the end of the run. Default False
	OUTPUTS:
		wStar 	: eigenvalue that the solver converged to
		vStar 	: eigenvector that the solver converged to
		conIss 	: (optional) list that logs any issues with non-convergence, only returned if conLog is True
	'''
	#first, figure out what size vectors we are dealing with!
	n = np.shape(u)[0]
	
	###GOT TO HERE - NEED TO MAKE THE SOLVING FUNCTION ENTIRELY REAL, SO TURN IT INTO A 2N VECTOR OF REAL VALUES, THEN CAST IT BACK AT THE END :l SEE https://stackoverflow.com/questions/21834459/finding-complex-roots-from-set-of-non-linear-equations-in-python OR SIMILAR
	
	#create lambda function that we will pass to fsolve in each loop of the iteration.
	fsolveFn = lambda x,w,v: np.matmul(M(w,theta),x) - np.matmul(Mprime(w,theta),v)
	#annoyingly, scipy cannot deal with complex valued equations, so we need a wrapper for this function which outputs real arrays. As such, the following function outputs a (2n,) vector of real values corresponding to the real and imaginary parts of the equation.
	def fRealFn(z,w,v):
		'''
		z should be a (2n,) numpy array which we convert to a complex-valued (n,) array, pass into fsolveFn, then cast the result back to a (2n,) real valued array.
		For internal use only, not to be seen externally.
		'''

		x = RealToComp(z) #cast back to complex
		fComplexValue = fsolveFn(x,w,v) #evaluate
		realOut = CompToReal(fComplexValue) #cast back to real array...
		
		return realOut
	
	#storage for iteration outputs
	wStore = np.zeros((maxItt+1), dtype=complex); wStore[0] = w0
	vStore = np.zeros((n,maxItt+1), dtype=complex); vStore[:,0] = v0	
	errStore = np.zeros((maxItt+1), dtype=float); errStore[0] = tol + 1 #auto-fail to start 1st iteration!
	#iteration counter
	currItt = 1
	
	while currItt<=maxItt and errStore[currItt-1]>tol:
		#whilst we have no exceeded the maximum number of iterations and the current tolerance in the solution is too high
		
		#solve M(w_k)x = M'(w_k)v_k for x; but we need to use z=real,imag (x) because SciPy
		func = lambda z: fRealFn(z,wStore[currItt-1],vStore[:,currItt-1])
		#for want of a better guess, use the current "eigenvector" as a guess of the solution...
		z = fsolve(func, CompToReal(vStore[:,currItt-1]))
		x = RealToComp(z) #cast back to complex array to save variables
		
		#set eigenvalue; w_{k+1} = w_k - u*v_k/u*x
		wStore[currItt] = wStore[currItt-1] - np.vdot(u,vStore[:,currItt-1])/np.vdot(u,x)
		
		#normalise eigenvalue
		vStore[:,currItt] = x/norm(x)
		
		#compute error, ||M(w_k+1)v_k+1||_2 should be small
		errStore[currItt] = norm(np.matmul(M(wStore[currItt],theta),vStore[:,currItt]))
		
		#incriment counter
		currItt += 1

	#dump values for manual investigation
	conIss = []
	#warnings that we might encounter
	if currItt>=maxItt:
		warn('Solver reached iteration limit')
		conIss.append('MaxItter')
	else:
		if talkToMe:
			print('Solver stopped at iteration %d' % (currItt-1))
		#shave off excess storage space to save some memory in this case
		#remember slicing in python is a:b goes from index a through to b-1 inclusive!
		wStore = wStore[0:currItt]
		vStore = vStore[:,0:currItt]
		errStore = errStore[0:currItt]
	if errStore[currItt-1]>tol:
		warn('Solver did not find a solution to the required tolerance: needed %.2e but only reached %.5e' % (tol, errStore[-1]))
		conIss.append('NoConv')
	else:
		if talkToMe:
			print('Difference from zero in converged eigenpair: %.5e' % errStore[currItt-1])
	if np.abs(norm(vStore[:,currItt-1])-1)>=tol:
		warn('Eigenvector is approximately 0')
		conIss.append('ZeroEVec')
	
	if conLog:
		#if we want the massive error log we need to return it here
		conIss.append(wStore)
		conIss.append(vStore)
		conIss.append(errStore)
		return wStore[currItt-1], vStore[:,currItt-1], conIss
	#otherwise, we just return the "answer"
	return wStore[currItt-1], vStore[:,currItt-1]

#for one fixed value of the quasimomentum, find me some eigenvalues :)
def SweepQM(G, nSamples, theta=np.zeros((2,), dtype=float), wRange=np.asarray([0, 2*np.pi]), maxItt=300, tol=1e-8):
	'''
	At the fixed value of the quasimomentum, find eigenvalues of the graph problem defined by G by starting the NLII at nSamples of w0 and recording the unique values it converges to.
	INPUTS:
		G 	: Graph instance, defining the problem that we wish to solve. G.ConstructM() should be callable
		nSamples 	: int, nunber of starting values w0 to begin the NLII at
		theta 	: (optional) (2,) numpy array, value of the quasimomentum for this sweep. Default [0,0]
		wRange 	: (optional) (2,) numpy array, the lower and upper bounds to sample w0 - these should be set to one period of the M matrix. Note that the returned eigenvalues can be outside this range. Default [0,2*np.pi]
		maxItt 	: (optional) int, max number of iterations of the NLII method to perform. Default 300.
		tol 	: (optional) float, solution tolerance of the NLII method. Default 1e-8.
	OUTPUTS:
		eVals 	: numpy array, unique eigenvalues that were found by the solver during this sweep
		eVecs 	: list of numpy arrays, i-th member of the list is a numpy array whose columns are the eigenvectors corresponding to the eigenvalue eVals[i]
	'''
	#create w0 samples to sweep through
	w0Samples = np.linspace(wRange[0], wRange[1], num=nSamples, endpoint=False)
	#create lists to store eigenvalues and eigenvectors we find
	eValList = []
	eVecList = []
	#create function handles for the M matrix and it's derivative
	M, Mprime = G.ConstructM(dervToo=True)
	#solver initial data
	v0 = UnitVector(0,G.nVert)
	u = UnitVector(0,G.nVert)
	
	#begin sweep
	for i in range(nSamples):
		wStar, vStar, conIss = NLII(M, Mprime, v0, u, w0=w0Samples[i], theta=theta, maxItt=maxItt, tol=tol)
		#if conIss contains issues we don't trust the value put out, otherwise append it to the lists :)
		if len(conIss)<=3:
			#no extra issues were thrown with the convergence - record e'val and e'vector
			eValList.append(wStar)
			eVecList.append(vStar)
		else:
			#if there was an issue with convergence, we should silently do nothing...
			pass
	
	#we should now have a list of candidate eigenvalues and eigenvectors. Now we need to remove dupicates.
	if len(eValList)!=0:
		eVals, uIDs = RemoveDuplicates(eValList)
		#we now have the unique eigenvalues, but need to associate the eigenvectors too
		vecArray = np.asarray(eVecList).T
		eVecs = []
		for i in range(np.max(uIDs)):
			eVecs.append( vecArray[:,uIDs==(i+1)] )
	else:
		#no eigenvalues found in this search range, return array and vector of zeros
		warn('No eigenvalues found for QM [%.8f, %.8f]' % (theta[0], theta[1]))
		eVals = np.asarray([0])
		eVecs = u;	eVecs[0] = 0; #zero vector for eigenvector
	
	return eVals, eVecs

#for finding the spectrum of a quantum graph problem by brute force - loop over a QM grid and a w grid...
def FindSpectrum(G, qmSamples, wSamples, tRange=np.asarray([-np.pi, np.pi]), wRange=np.asarray([0, 2*np.pi]), vecsToo=False, qmSymmetry=False):
	'''
	Find the spectrum (and associated eigenvectors if we wish) of the quantum graph problem stored in G by meshing the quasimomentum range, and for each value finding the eigenvalues within a given range. The "whole" spectrum is then assembled from the union of the eigenvalues found for each QM value.
	INPUTS:
		G 	: Graph instance, defining the problem that we wish to solve. G.ConstructM() should be callable
		qmSamples 	: int, number of samples for the quasimomentum to use
		wSamples 	: int, number of starting values for the NLII solver, for each QM  value. Passed to SweepQM() function.
		tRange 	: (optional) (2,) numpy array, range for the two quasimomentum coefficients. Default [-np.pi,np.pi]
		wRange 	: (optional) (2,) numpy array, the lower and upper bounds to sample w0 - these should be set to one period of the M matrix. Returned eigenvalues will be in the range - those found outside this range will be discarded. Default [0,2*np.pi]
		vecsToo 	: (optional) bool, if True then eigenvectors will be returned alongside the eigenvalues that were found. Default False
		qmSymmetry 	: (optional) bool, if True then the problem we are solving is symmetric in the two quasimomentum parameters, so it is necessary to sweep through only one of them rather than both. Default False
	OUTPUTS:
		spectrum 	: complex numpy array, eigenvalues that were computed. Although we know that they should all be real as M is Hermitian, we need to return complex values for python not to complain too much.
		vecList 	: (optional) list of numpy arrays, i-th member of the list is a numpy array whose columns are the eigenvectors corresponding to the eigenvalue spectrum[i]
	'''
	
	qmSpace = np.linspace(tRange[0], tRange[1], num=qmSamples, endpoint=True)
	spectrum = np.asarray([]) #initalise store for spectrum
	if vecsToo:
		vecList = [] #initialise list of eigenvectors
	
	#check for symmetry to try to avoid nested loops
	if qmSymmetry:
		for i in range(qmSamples):
			#get eigenvalues for this value of the QM, leaving the second QM value as 0
			qmEVals, qmEVecs = SweepQM(G, wSamples, theta=np.asarray([qmSpace[i],0.0]), wRange=wRange)
			#deal with "out of range" eigenvalues
			inRangeVals = (qmEVals < wRange[1]) & (qmEVals > wRange[0])
			if vecsToo:
				#filter the list for undesirable eigenvectors
				filteredVecList = []
				for i in range(np.shape(qmEVals)[0]):
					if inRangeVals[i]:
						#keep this set of eigenvectors
						filteredVecList.append(qmEVecs[i])
				vecList = vecList + qmEVecs #concatenating lists using + is fine...
			#now filter undesirable eigenvalues
			qmEVals = qmEVals[inRangeVals]
			#append the data we found to our existing stores
			spectrum = np.hstack( (spectrum, qmEVals) )				
	else:
		#we cannot assume QM symmetry. We are required to work with a nested loop
		for i in range(qmSamples):
			for j in range(qmSamples):
				#QM = [qmSpace[i], qmSpace[j]], IE all possible pairs of values taken from qmSpace
				#deal with "out of range" eigenvalues
				inRangeVals = (qmEVals < wRange[1]) & (qmEVals > wRange[0])
				if vecsToo:
					#filter the list for undesirable eigenvectors
					filteredVecList = []
					for i in range(np.shape(qmEVals)[0]):
						if inRangeVals[i]:
							#keep this set of eigenvectors
							filteredVecList.append(qmEVecs[i])
					vecList = vecList + qmEVecs #concatenating lists using + is fine...
				#now filter undesirable eigenvalues
				qmEVals = qmEVals[inRangeVals]
				#append the data we found to our existing stores
				spectrum = np.hstack( (spectrum, qmEVals) )				
	#once we are here, we should have assembled the complete spectrum to the accuracy of the discretisation of our QM domain.
	if vecsToo:
		return spectrum, vecList
	
	return spectrum

#for removing duplicates where "duplicates" means "close enough together to probably be the same"
def RemoveDuplicates(valIn, tol=1e-8):
	'''
	Remove duplicate entries from a given list or numpy array of complex values; classing duplicates to be be those values whose norm difference is less than a given tolerance.
	INPUTS:
		valIn 	: list or 1D numpy array, complex values to be sorted for duplicates
		tol 	: (optional) float, tolerance within which values are classed as duplicates
	OUTPUTS:
		unique 	: complex numpy array, the unique values using the mean of all values determined to be "the same"
		uIDs 	: int numpy array, if vals[i] is deemed a duplicate of vals[j], then uIDs[i]=uIDs[j], essentially returning the "groupings" of the original values.
	'''
	
	vals = np.asarray(valIn) #in case the input is a list, this will allow us to do logic slicing with numpy arrays
	nVals = np.shape(vals)[0]
	uIDs = np.zeros((nVals,), dtype=int)
	newID = 1
	
	for currInd in range(nVals):
		#has this value already be assigned a group?
		if uIDs[currInd]==0:
			#this value does not have a group yet - give it one, then go through and find it's other members
			uIDs[currInd] = newID
			newID += 1 #update the group number that a new group would take
			for checkInd in range(currInd+1, nVals):
				#go along the array sorting values into this group
				if uIDs[checkInd]==0 and norm(vals[currInd]-vals[checkInd])<tol:
					#this value is also not in a group yet and is a "duplicate" of the ungrouped element we are considering - they belong in the same group
					uIDs[checkInd] = uIDs[currInd]
			#we have made a new group of values and found all it's members... good job.
			currInd += 1

	#now uIDs is a grouping of all the values that we needed, and max(uIDs) is the number of unique values that we found
	unique = np.zeros((np.max(uIDs),), dtype=complex)
	for i in range(np.max(uIDs)):
		#average the values in group i and put them into the unique array
		unique[i] = np.mean( vals[uIDs==(i+1)] )
			
	return unique, uIDs

#help (and reminder) as to how to use the Graph class that I have assembled. In particular, how to create instances of graphs using it and the correct formats for the inputs (hint: they're silly).
def GraphConstructorHelp():
	'''
	Instances of the Graph class are intended as setups for one particular quantum graph that we wish to solve an ODE on (the edges of).
	Presently; the program cannot handle graphs with internal loops or curved edges, or graphs which contain vertices that have 3 or more edges between them. The program will interpret vertices with two edges connecting them as reflecting the (quasi-) periodic boundary conditions imposed on the graph system, and due to the restriction of straight edges will assume that both edges have the same length. If this is not the case in the desired system, consider adding a "dummy" vertex between to break up the longer of the two edges.
	
	To create an instane of the Graph class, one uses the command
		G = Graph(fileName)
	providing the path to the file fileName.
	
	fileName expects a .csv file in the following format:
		Vertices
		vID, 	x, 		y,
		vID2, 	x2, 	y2, 
		...
		Edges
		leftID, 	rightID, 	length, 	theta1, 	theta2,
		leftID2, 	rightID2, 	length2, 	theta1, 	theta2,
		...
		Coupling
		vID, 	coupCoeffs ...
		...
	After the Vertices keyword, the ID numbers of each vertex should be given starting from 0 and in sequential order. Each line consists of:
		vertex's ID (vID),
		it's x-co-ordinate (x),
		and y-co-ordinate (y)
	This pattern continues line-by-line, each line introducing a new vertex, until the Edges keyword is reached.
	After the Edges keyword, each line encodes an edge between vertices with the vertices with leftID and rightID according to:
		the vertex at the left of the edge by ID (leftID),
		the vertex at the right of the edge by ID (rightID),
		the length of the connecting edge (length), [this field can be left blank, and the program will replace it with Euclidean distance]
		the coefficient along the edge for the 1st quasimomentum parameter (theta1),
		the coefficient along the edge for the 2nd quasimomentum parameter (theta2)
	This pattern continues until the Coupling keyword is reached.
	After the Coupling keyword, each line encodes the coupling constant at the given vertex ID, according to:
		the vertex at which the coupling occurs (vID)
		the form of the coupling constant at the vertex (coupCoeffs ...). Any number of coefficients can be provided and they form a coupling constant of the form c0 + c1*w + c2*w^2 + ... IE a polynomial coupling in powers of w. If a vertex does not have a coupling constant provided, it is assumed to be zero.

	A graph is then created with the following attributes:
		
	vPos is a (2,n) shape numpy array; the columns of which should be the (x,y) positions of the vertices of the graph. The length (read: number of columns) of vPos will be interpretted as the number of vertices in the graph.
	
	adjMat is an (n,n) shape numpy array; being similar to the adjacency matrix for the graph that is to be constructed. However the direction of the edges of graph must also be encoded, along with how many edges we have connecting each pair of vertices. As such the matrix passed in adjMat must have the following structure:
		 - adjMat must have zeros on the leading diagonal (a warning will be thrown if this is not the case, but construction will then continue assuming zeros were meant on the diagonal).
		 - adjMat[i,j] for i<j (in the upper triangle of adjMat) should be either 0 or 1. A 1 represents that vertex i is connected to vertex j with vertex i the "left" endpoint of the edge and vertex j the "right". A 0 represents that there is no edge between i and j with this orientation.
		 - adjMat[i,j] for i>j (in the lower triangle of adjMat) should be either 0 or 1. A 1 represents that vertex i is connected to vertex j with vertex i the "right" endpoint of the edge and vertex j the "left". A 0 represents that there is no edge between i and j with this orientation.
	 As such the matrix adjMat need not be symmetric, and in general will not be.
	 
	 lenMat is a dictionary whose keys are tuples (j,k). Only tuples (j,k) for which adjMat[j,k]!=0 are keys, as the others will not need to be accessed nor store anything. lenMat[j,k] is a numpy array whose shape is (adjMat[j,k],) - IE it stores as many lengths as there are edges between vertices j and k. These are stored in a manner consistent with theMat, and with the construction method for the M matrix.
	 
	 theMat is a dictionary whose keys are tuples (j,k). It functions like lenMat, but stores a (2, adjMat[j,k]) numpy array at each key - here the columns of this array correspond to sets of coefficients of the quasimomentum parameters, one set for each edge between the vertices.
	 
	 coupConsts is a function that returns the value of the (diagonal) matrix of coupling constants at a given value of w. It is constructed automatically along with the other variables that the Graph needs.
	'''
	help(GraphConstructorHelp) #laziness is a virtue...
	
	return