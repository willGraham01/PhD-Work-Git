#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 12:34:05 2019

@author: will

This file will serve to create the datastructures needed for the quantum graph numerics.
In particular, it will setup the vertex class and provide methods for assembling a graph consisting of vertices.

Presently we are still assuming we have -d^2/dx^2 on each vertex still (subject to quasimomentum shifts, of course).
"""

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
	Attributes:
		nVert 	: int, number of vertices in the graph
		vPos 	: (2,n) numpy array, each column is the (x,y) position of one of the vertices in the graph. By default, vertices will have ID's from 0 to nVert-1.
		adjMat 	: (nVert,nVert) numpy array, the "modified" adjacency matrix for the graph that also encodes directions
		lenMat 	: (nVert,nVert) numpy array, stores the lengths of connected vertices to save recomputation
		theMat 	: (nVert,nVert,2) numpy array, stores the coefficents of the two components of theta for each edge. That is, sum(theMat[j,k,:]*[theta1, theta2]) = theta_jk.
	Methods:
	'''
	#Default attribute values for a graph
	nVert = 1
	vList = np.asarray([])
	adjMat = np.zeros((nVert,nVert))
	lenMat = np.zeros((nVert,nVert))
	theMat = np.zeros((nVert,nVert,2))
	
	def __init__(self, vPos, adjMat, theMat=np.zeros((1,1))):
		'''
		Construction method for the Graph class. An instance of the Graph class will be created with its attributes generated to be consistent with the inputs.
		INPUTS:
			vPos 	: (2,n) numpy array, each column is an (x,y) vertex position 
			adjMat 	: (nVert,nVert) numpy array, the "modified" adjacency matrix for the graph that also encodes directions - use the command GraphConstructorHelp() included in this module for more information.
			theMat 	: (n,n,2) numpy array - optional. This matrix contains the quasimomentum coefficients on each edge. That is, theMat[j,k,:]*theta = theta[j,k]. By default these are set to 0 everywhere.
		'''
		
		self.vPos = np.copy(vPos) #these are the vertex positions
		self.nVert = np.shape(vPos)[1] #vPos of shape (2,n) implies there are n vertices
		self.adjMat = np.copy(adjMat) #modified adjacency matrix is given as input
		self.lenMat = np.zeros_like(adjMat, dtype=float) #now create the matrix of lengths
		
		if np.sum(np.diag(adjMat))>0:
			#diagonal entries are not zero - graph may have loops?
			warn('adjMat has non-zero diagonal entries. Construction will continue assuming these are meant to be zero; and will be recorded as 0 in G.adjMat')
			np.fill_diagonal(self.adjMat, 0)
		if np.sum(theMat)==0:
			#either no theMat provided, or there is no quasimomentum dependance. Either way, ensure that what we are storing is the correct size
			self.theMat = np.zeros((self.nVert,self.nVert,2))
		elif np.shape(theMat)!=(self.nVert,self.nVert,2):
			#a custom theMat has been provided, but the shape is incorrect
			raise ValueError('theMat has incorrect shape: expected (%d, %d, 2) but got ' % (self.nVert, self.nVert), np.shape(theMat))
		else:
			#custom theMat provided of the correct shape. Assign
			self.theMat = np.copy(theMat)
		
		for i in range(self.nVert):
			 for j in range(i,self.nVert):
				 #update the distance matrix. Note that the distances are symmetric.
				 self.lenMat[i,j] = norm(vPos[:,i]-vPos[:,j])
				 self.lenMat[j,i] = norm(vPos[:,i]-vPos[:,j])

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
					#distances are reflexive - so take the lengths from lenMat after combining the two connection lists
					lens = self.lenMat[k,lCon+rCon]
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
							kRight.append(self.lenMat[j,k])
							t1Right.append(self.theMat[j,k,0])
							t2Right.append(self.theMat[j,k,1])
							if self.adjMat[k,j]>0:
								#there is also a reverse connection, so we need to append to the left lists too
								kLeft.append(self.lenMat[k,j])
								t1Left.append(self.theMat[k,j,0])
								t2Left.append(self.theMat[k,j,1])
						else:
							#connection is in the lower triangle, so j is the right vertex and k is the left vertex
							kLeft.append(self.lenMat[j,k])
							t1Left.append(self.theMat[j,k,0])
							t2Left.append(self.theMat[j,k,1])
							if self.adjMat[k,j]>0:
								#there is also a reverse connection, so we need to append to the right lists too
								kRight.append(self.lenMat[k,j])
								t1Right.append(self.theMat[k,j,0])
								t2Right.append(self.theMat[k,j,1])
					else: 
						# self.adjMat[k,j]>0, but there is no reverse connection as it would have been caught in the previous case
						if k<j:
							#connection is in the upper triangle, so k is the left vertex and j is the right vertex
							kLeft.append(self.lenMat[k,j])
							t1Left.append(self.theMat[k,j,0])
							t2Left.append(self.theMat[k,j,1])
						else:
							#connection is in the lower triangle, so k is the right vertex and j is the left vertex
							kRight.append(self.lenMat[k,j])
							t1Right.append(self.theMat[k,j,0])
							t2Right.append(self.theMat[k,j,1])
					
					kLeft = np.asarray(kLeft)
					kRight = np.asarray(kRight)
					t1Left = np.asarray(t1Left)
					t2Left = np.asarray(t2Left)
					t1Right = np.asarray(t1Right)
					t2Right = np.asarray(t2Right)
					masterList[j][k] = [kLeft, t1Left, t2Left, kRight, t1Right, t2Right]
					
					#in the off-diagonal terms, masterList stores a further list of 6 entries; effectively in two groups. The first 3 are a list of the lengths of edges with vertex k on the left, followed by the QM parameters in for the first and second components in the next two. The final 3 are similar, but for edges with k on the right.
		
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
	
	def Draw(self, offSet=1, tex=True, show=False):
		'''
		Draws the structure of the graph as a matplotlib figure.
		INPUTS:
			offSet 	: (optional) int, number to add to each of the vertex IDs for consistency between program and analytic problem. Default 1
			tex 	: (optional) bool, if True then the diagram will be plotted using LaTeX rendering, otherwise use the matplotlib defaults. Default True
			show 	: (optional) bool, if True then the diagram is displayed by the method before returning the outputs - use for suppression of plots in loops. Default False
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
		#now let's make a list of all the arrows we want to draw :)
		for i in range(self.nVert):
			vi = (self.vPos[0,i],self.vPos[1,i])
			#save time looping by not revisiting vertices we've already drawn
			for j in range(i+1,self.nVert):
				vj = (self.vPos[0,j],self.vPos[1,j])
				if self.adjMat[i,j]>0 and self.adjMat[j,i]>0:
					#2-way connection, draw two arrows
					arrow_ij = pts.FancyArrowPatch(vi, vj, connectionstyle="arc3,rad=0.5",**kw)
					arrow_ji = pts.FancyArrowPatch(vj, vi, connectionstyle="arc3,rad=0.5",**kw)
					arrowList.append(arrow_ij)
					arrowList.append(arrow_ji)
				elif self.adjMat[i,j]>0:
					#only one connection. As j>=i+1, we are in the upper triangle so vi is left and vj is right. Make sure the arrow points this way, and it doesn't need to be curved
					arrow_ij = pts.FancyArrowPatch(vi, vj, **kw)
					arrowList.append(arrow_ij)
				elif self.adjMat[j,i]>0:
					#only one direction. As j>i+1, we are in the lower triangle so vj is left and vi is right. Make sure the arrow points this way, and it doesn't need to be curved
					arrow_ji = pts.FancyArrowPatch(vj, vi, **kw)
					arrowList.append(arrow_ji)
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

#nonlinear inverse interation solver...
def NLII(M, Mprime, v0, u, w0=np.pi, theta=np.zeros((2), dtype=float), maxItt=100, tol=1.0e-8, conLog=True):
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

#help (and reminder) as to how to use the Graph class that I have assembled. In particular, how to create instances of graphs using it and the correct formats for the inputs (hint: they're silly).
def GraphConstructorHelp():
	'''
	Instances of the Graph class are intended as setups for one particular quantum graph that we wish to solve an ODE on (the edges of).
	Presently; the program cannot handle graphs with internal loops or curved edges, or graphs which contain vertices that have 3 or more edges between them. The program will interpret vertices with two edges connecting them as reflecting the (quasi-) periodic boundary conditions imposed on the graph system, and due to the restriction of straight edges will assume that both edges have the same length. If this is not the case in the desired system, consider adding a "dummy" vertex between to break up the longer of the two edges.
	
	To create an instane of the Graph class, one uses the command
		G = Graph(vPos, adjMat)
	providing the two arguments vPos and adjMat.
	
	vPos is a (2,n) shape numpy array; the columns of which should be the (x,y) positions of the vertices of the graph. The length (read: number of columns) of vPos will be interpretted as the number of vertices in the graph.
	
	adjMat is an (n,n) shape numpy array; being similar to the adjacency matrix for the graph that is to be constructed. However the direction of the edges of graph must also be encoded, along with how many edges we have connecting each pair of vertices. As such the matrix passed in adjMat must have the following structure:
		 - adjMat must have zeros on the leading diagonal (a warning will be thrown if this is not the case, but construction will then continue assuming zeros were meant on the diagonal).
		 - adjMat[i,j] for i<j (in the upper triangle of adjMat) should be either 0 or 1. A 1 represents that vertex i is connected to vertex j with vertex i the "left" endpoint of the edge and vertex j the "right". A 0 represents that there is no edge between i and j with this orientation.
		 - adjMat[i,j] for i<j (in the lower triangle of adjMat) should be either 0 or 1. A 1 represents that vertex i is connected to vertex j with vertex i the "right" endpoint of the edge and vertex j the "left". A 0 represents that there is no edge between i and j with this orientation.
	 As such the matrix adjMat need not be symmetric, and in general will not be.
	 
	 
	 theMat is an (n,n,2) numpy array and is optional. If theMat is not provided or is provided an is identically zero, this corresponds to a problem without quasimomentum dependance (IE a non-periodic medium). Otherwise, each (j,k,:) position contains a 2-vector of the coefficients for the quasimomentum parameters. That is to say, if theta is the quasimomentum vector and theMat[j,k] = [t1,t2] then we are solving -(d/dx + i[t1,t2].*theMat[j,k])^2 u = w^2 u on the edge j,k.
	'''
	
	help(GraphConstructorHelp)
	
	return

#functions for testing the construction of the M-matrix and the eigenvalues that it spits out
def TestVars():
	
	#Test variables for the TFR problem
	posTFR = np.asarray( [ [0.5, 1.0, 0.5], [1.0, 0.5, 0.5] ] )
	adjMatTFR = np.asarray( [ [0, 0, 1], [0, 0, 1], [1, 1, 0] ] )
	#when building this, remember that python indexes from 0 not 1
	theMatTFR = np.zeros((3,3,2))
	theMatTFR[0,2,:] = np.asarray([0.0,1.0]) #v1 is vertically above v3... so theta_2 should be here
	theMatTFR[2,0,:] = np.asarray([0.0,1.0]) #symmetry
	theMatTFR[1,2,:] = np.asarray([1.0,0.0]) #v2 is right of v3
	theMatTFR[2,1,:] = np.asarray([1.0,0.0]) #symmetry
		
	G_TFR = Graph(posTFR, adjMatTFR, theMatTFR)
	M_TFR = G_TFR.ConstructM()
	
	#Test variables for the problem in the EKK paper
#	posEKK = np.asarray( [ [], [] ] )
#	adjMatEKK = np.asarray( [ [], [], [], [] ] )
#	theMatEKK = np.zeros((4,4,2))
#	#THEMAT STUFF HERE
#	G_EKK = Graph(posEKK, adjMatEKK, theMatEKK)
#	M_EKK = G_EKK.ConstructM()
	
	return G_TFR, M_TFR, posTFR, adjMatTFR#, G_EKK, M_EKK, posEKK, adjMatEKK

def TFR_Exact(w, theta=np.zeros((2))):
	#exact M-matrix for the TFR problem
	
	mat = np.zeros((3,3), dtype=complex)
	mat[0,0] = -2*w*cot(w/2)
	mat[1,1] = -2*w*cot(w/2)
	mat[2,2] = -4*w*cot(w/2)
	mat[0,2] = 2*w*np.cos(theta[1]/2)*cosec(w/2)
	mat[2,0] = 2*w*np.cos(theta[1]/2)*cosec(w/2)
	mat[1,2] = 2*w*np.cos(theta[0]/2)*cosec(w/2)
	mat[2,1] = 2*w*np.cos(theta[0]/2)*cosec(w/2)
	
	return mat

def CompareConstructions(exact, computational, nSamples=1000, theta1Present=True, theta2Present=False, wRange=np.asarray([-np.pi,np.pi]), thetaRange=np.asarray([-np.pi,np.pi]), tol=1e-8):
	'''
	Given two function handles, exact and computational, performs nSamples evaluations of both functions over prescribed intervals to determine if the computationally constructed solution and the exact form of the M-matrix agree.
	INPUTS:
		exact 	: lambda function, given (w,theta) produces the M-matrix from an analytic expression
		computational 	: lambda function, given (w,theta) produces the M-matrix from computational construction
		nSamples 	: (optional) int, number of samples in each interval - total samples is nSamples^3
		theta1Present 	: (optional) bool, if true then we will sample theta1 values too - if false then we assume theta1=0. Default True
		theta2Present 	: (optional) bool, if true then we will sample theta2 values too - if false then we assume theta2=0. Default False
		wRange 	: (optional) (2,) numpy array, interval of w to test. Default [-pi, pi]
		thetaRange 	: (optional) (2,) numpy array, interval of theta to test. Default [-pi,pi]
		tol 	: (optional) float, tolerance of Frobenius norm to accecpt as numerical error
	OUTPUTS:
		failList 	: list, each entry in the list is a further list of 2 members: the w and theta values of the failures for the exact solution to match the computational one
	'''
	#sampled calues of w
	wVals = np.linspace(wRange[0],wRange[1],num=nSamples,endpoint=True)
	
	#sampled values of theta
	tVals = np.zeros((2,nSamples))
	if theta1Present:
		tVals[0,:] = np.linspace(thetaRange[0],thetaRange[1],num=nSamples,endpoint=True)
	if theta2Present:
		tVals[1,:] = np.linspace(thetaRange[0],thetaRange[1],num=nSamples,endpoint=True)
	
	matches = 0
	fails = 0
	failList = []
	nTrials = nSamples
	
	for i in range(nSamples):
		if (theta1Present and theta2Present):
			nTrials = nSamples*nSamples*nSamples
			for j in range(nSamples):
				for k in range(nSamples):
					e = exact(wVals[i], tVals[[0,1],[j,k]])
					c = computational(wVals[i], tVals[[0,1],[j,k]])
			
					#Frobenius norm
					froNorm = norm(e-c,'fro')		
					#determine mismatch cases
					if froNorm>=tol:
						#too much mismatch
						fails += 1
						failList.append([wVals[i], tVals[[0,1],[j,k]]]) #append the fail points
					else:
						#pass test
						matches += 1
		elif theta1Present or theta2Present:
			#only looping over theta1 values
			nTrials = nSamples*nSamples
			for j in range(nSamples):
				e = exact(wVals[i], tVals[:,j])
				c = computational(wVals[i], tVals[:,j])
		
				#Frobenius norm
				froNorm = norm(e-c,'fro')		
				#determine mismatch cases
				if froNorm>=tol:
					#too much mismatch
					fails += 1
					failList.append([wVals[i], tVals[:,j]]) #append the fail points
				else:
					#pass test
					matches += 1
		else:
			#only looping over w values
			e = exact(wVals[i], tVals[:,j])
			c = computational(wVals[i], tVals[:,j])
	
			#Frobenius norm
			froNorm = norm(e-c,'fro')		
			#determine mismatch cases
			if froNorm>=tol:
				#too much mismatch
				fails += 1
				failList.append([wVals[i], tVals[:,j]]) #append the fail points
			else:
				#pass test
				matches += 1		
						
	print('Test result:')
	print('Passes: %d ( %.3f )' % (matches, float(matches)/nTrials))
	print('Fails: %d ( %.3f )' % (fails, float(fails)/nTrials))
	print('Returning list of failure cases')
	
	return failList

#delete once complete, but for now just auto give me the test variables
#G_TFR, M_TFR, vTFR, aTFR, G_EKK, M_EKK, vEKK, aEKK = TestVars()
G_TFR, M_TFR, vTFR, aTFR = TestVars()