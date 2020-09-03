#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:01:39 2020

@author: will

File containing the Graph class that is required for solving spectral Quantum Graph problems using the M-matrix.
Dependencies:
	VertexClass.py
	EdgeClass.py
	
Presently we are still assuming we have -d^2/dx^2 on each vertex still (subject to quasimomentum shifts, of course).
"""

import numpy as np
from numpy import cos, sin

import matplotlib.pyplot as plt
import matplotlib.patches as pts #for arrows
from matplotlib import rc #tex in figures

from VertexClass import Vertex
from EdgeClass import Edge
from AuxMethods import csc, cot

class Graph:
	'''
	Initalise with G = Graph(V, E).
	An instance of the Graph class enables us to assemble the M-matrix (and it's derivative) as a function for a given embedded graph.
	The constructor method should be passed a list of Vertex objects V and a list of Edge objects E.
	ATTRIBUTES:
		_nVert: 	int, the number of vertices in the graph
		_vDict: 	dict of Vertex objects, the vertices in the graph with Vertex vID at key vID in the dict.
		_eDict: 	dict of lists of Edge objects, at key (vIDLeft, vIDRight) there is a list of edge objects containing each edge directed from vIDLeft to vIDRight
		_idTranslation: 	dict, each key is an integer corresponding to a vID in the list of vertices V given on initialisation. _idTranslation[vID] is the vID that this vertex was reassigned by the constructor, to ensure a sequential set of vertex labels starting at 0
		_adjMat: 	(_nVert, _nVert) int numpy array, the modified adjacency matrix. The (i,j)th entry of the matrix is the number of edges directed from (internal) vID i (left) --> (internal) vID j (right)
		_cConsts: 	(_nVert, ) float numpy array, the coupling constants for the vertices in this graph, ordered by internal ID
	METHODS:
		OriginalID: 	obtains the original ID of a vertex in the graph G
		InternalID: 	obtains the internal ID of a vertex provided to the initialisation method
		VPosMatrix: 	returns a matrix containing the positions of the vertices of G
		AdjMat: 	returns (a varient of) the adjacency matrix of G
		CConts: 	returns either a vector or diagonal matrix of the coupling constants for G
		Draw: 	draws the graph as a matplotlib figure
		ConstructM: 	constructs the Weyl-Tischmarsh function M for the graph G, returning a function handle
		ConstructAltM: 	constructs the M-matrix with a prefactor removed, in an attempt to handle a computationally nicer function
	'''
	
	def __init__(self, V, E):
		'''
		Initialisation method for Graph class.
		INPUTS:
			V: 	list of Vertex objects, the vertices in the Graph
			E: 	list of Edge objects, the edges of the Graph
		Vertices are given new IDs corresponding to their order of occurance in the list V
		'''
		
		self._vDict = {}
		self._eDict = {}
		self._idTranslation = {}
		
		#first, we create a temporary dictionary that maps the vIDs of the input vertices onto some nicer, sequential IDs
		vIDNext = 0
		for v in V:
			self._idTranslation[v.vID] = vIDNext
			vIDNext += 1
		self._nVert = vIDNext
		#failsafe: check that no two vertices had the same index
		if self._nVert != len(self._idTranslation):
			#there are fewer vertices in V than are stored in our dictionary, return an error
			raise RuntimeError('Vertices do not have unique IDs, construction failed')		

		#now we know the number of vertices we can construct our directed adjacency matrix and coupling constant vector
		self._adjMat = np.zeros((self._nVert, self._nVert), dtype=int)
		self._cConsts = np.zeros((self._nVert), dtype=float)
			
		#place into internal vertex dictionary, with translated ids
		for v in V:
			internalID = self._idTranslation[v.vID]
			self._vDict[internalID] = Vertex(internalID, v.vPos, v.coupConst)
			self._cConsts[internalID] = v.coupConst

		#contstruct internal edge dictionary, with translated ids
		for e in E:
			#translated vertex IDs for this edge
			transLeftID = self._idTranslation[e.leftV.vID]
			transRightID = self._idTranslation[e.rightV.vID]
			
			if (transLeftID, transRightID) in self._eDict:
				#if this key already exists in the dictionary, we are adding another edge between two vertices that already have one, so we need to append to the already created list at this key
				self._eDict[transLeftID, transRightID].append( Edge(self._vDict[transLeftID], self._vDict[transRightID], \
			  e.length, e.qm) )
			else:
				#these two edges were previously unconnected, so create a new key and list for them and add this connection
				self._eDict[transLeftID, transRightID] = [ Edge(self._vDict[transLeftID],  \
				self._vDict[transRightID], e.length, e.qm) ]
			#make a record of this edge in the adjacency matrix
			self._adjMat[transLeftID, transRightID] += 1
		
		return

	def __repr__(self):
		'''
		Default output of a call to a graph object.
		'''
		return "<Graph instance with %d vertices and %d edges>" % (self._nVert, np.sum(np.sum(self.AdjMat())))
		
	def __str__(self):
		'''
		Print text that is displayed for Graph objects when called with the print() function. Currently produces an image of the graph and returns some informative text
		'''
		self.Draw()
		
		return "Graph instance with %d vertices and %d edges" % (self._nVert, np.sum(np.sum(self.AdjMat())))

	def OriginalID(self, v):
		'''
		Return the original ID provided for this vertex, rather than the internal one that is used by the Graph object
		INPUTS:
			v: 	Vertex object, v must be a vertex of this graph
		OUTPUTS:
			oID: 	int, original vID of the vertex passed to the graph constructor
		'''
		
		oID = [orID for orID, vID in self._idTranslation.items() if vID == v.vID][0]
		#this should bea list of length 1 as there is a check in the construction method that no two of the original vertices had the same ID number, so we just take the only value of this list
		
		return oID
	
	def InternalID(self, v):
		'''
		Return the internal ID assigned to the vertex v that was passed to the graph constructor method
		INPUTS:
			v: 	Vertex object, v must be an element of the list V that was passed to Graph(V,E)
		OUTPUTS:
			iID: 	int, internal ID assigned to this vertex
		'''
		
		iID = self._idTranslation[v.vID]
		
		return iID
	
	def VPosMatrix(self):
		'''
		Creates a (2, self._nVert) float numpy array of all the vertex positions of the graph.
		Column vID contains the [x,y] position of the vertex with (internal) ID vID.
		INPUTS:
			
		OUTPUTS:
			vPosMat: 	(2, self._nVert) float numpy array, vertex positions of the graph
		'''
		
		vPosMat = np.zeros((2, self._nVert), dtype=float)
		for i, vi in self._vDict.items():
			vPosMat[:,i] = vi.vPos
		
		return vPosMat
	
	def AdjMat(self, binary=False, undirected=False):
		'''
		Return the adjacency matrix for this graph.
		INPUTS:
			binary: 	(optional) bool, if True then return the normal directed adjacency matrix (where a 1 indicates there is at least one connection between two edges). Default False.
			undirected: 	(optional) bool, if True then return the adjacency matrix of the graph without directions. If binary is True, this option is ignored. Default False.
		OUTPTS:
			aMat: 	(_nVert, _nVert) int numpy array, the requested adjacency matrix of the graph
		'''
		
		aMat = np.zeros((self._nVert, self._nVert), dtype=int)
		
		if binary:
			#1's as max values only
			aMat[self._adjMat>=1] = 1
		elif undirected:
			#don't care about directions either
			aMat[self._adjMat>=1] = 1
			for i in range(self._nVert):
				for j in range(self._nVert):
					aMat[i,j] = max([aMat[i,j], aMat[j,i]])
					aMat[j,i] = aMat[i,j]
		else:
			#essentially return _adjMat
			aMat = np.copy(self._adjMat)
		
		return aMat
	
	def CConsts(self, matrix=False):
		'''
		Returns the coupling constants assigned to the vertices of this graph.
		Coupling constants are placed into a numpy array and indexed by their internal IDs.
		INPUTS:
			matrix: 	(optional) bool, if True then the coupling constants are returned along the diagonal of an otherwise zero-entry diagonal matrix. Default False
		OUTPUTS:
			cConsts: 	(_nVert, ) float numpy array, containing the coupling constants of the vertices of the graph. If optional argument matrix is True, it will be of shape (_nVert, _nVert)
		'''
		
		if matrix:
			#return a diagonal matrix with cConsts on the diagonal
			return np.diag(self._cConsts)
		else:
			#return column vector that we already have
			return np.copy(self._cConsts)
		#shouldn't get to here, but it looks nice
		return
	
	def Draw(self, originalIDs=False, tex=True, show=False, radFac=0.25):
		'''
		Draws the structure of the graph as a matplotlib figure.
		INPUTS:
			offSet 	: (optional) bool, if True then the graph will be drawn using the IDs of the vertices that were passed to the constructor, otherwise it will use labels with the internal vIDs. Default False
			tex 	: (optional) bool, if True then the diagram will be plotted using LaTeX rendering, otherwise use the matplotlib defaults. Default True
			show 	: (optional) bool, if True then the diagram is displayed by the method before returning the outputs - use for suppression of plots in loops. Default False
			radFac 	: (optional) float, determines the "curviness" of the arrows in the displayed diagram. Smaller values result in straighter arrows and are useful for complex or cluttered graphs. Default 0.25
		OUTPUTS:
			fig, ax 	: matplotlib figure, axes handles, the assembled diagram of the graph.
		'''
		
		##pull out vertex positions into numpy array to make it nicer for matplotlib
		vPosMatrix = self.VPosMatrix()
		
		#setup axes
		xMin = min(np.min(vPosMatrix[0,:]),0.0); xMax = max(np.max(vPosMatrix[0,:]),1.0)
		yMin = min(np.min(vPosMatrix[1,:]),0.0); yMax = max(np.max(vPosMatrix[1,:]),1.0)
		#create axis instance
		fig = plt.figure();	ax = plt.gca()
		ax.set_xlim(xMin-0.1,xMax+0.1);	ax.set_ylim(yMin-0.1,yMax+0.1)
		ax.set_aspect(1)
		if tex:
			#tell python to write things out in LaTeX
			rc('text', usetex=True);	rc('font', family='serif')
			ax.xaxis.set_label_text('$x$');	ax.yaxis.set_label_text('$y$')
		else:
			rc('text', usetex=False)
			ax.xaxis.set_label_text('x');	ax.yaxis.set_label_text('y')
		ax.title.set_text('Diagram of graph')
		#define arrow style for connected edges
		style="Simple,tail_width=0.5,head_width=4,head_length=8"
		kw = dict(arrowstyle=style, color="r", zorder=1)
		
		#add the vertices to the plot, the manner in which this is done will depend on the options that have been specified above
		if tex and originalIDs:
			#use tex and the original vertex IDs
			for i, vi in self._vDict.items():
				oID = self.OriginalID(vi)
				ax.annotate('$v_'+str(oID)+'$', (vi.vPos[0],vi.vPos[1]), horizontalalignment='left', verticalalignment='bottom' )
		elif tex:
			#use tex but internal labels
			for i, vi in self._vDict.items():
				ax.annotate('$v_'+str(vi.vID)+'$', (vi.vPos[0],vi.vPos[1]), horizontalalignment='left', verticalalignment='bottom' )		
		elif originalIDs:
			#don't use tex but use original vertex IDs
			for i, vi in self._vDict.items():
				oID = self.OriginalID(vi)
				ax.annotate('v_'+str(oID), (vi.vPos[0],vi.vPos[1]), horizontalalignment='left', verticalalignment='bottom' )			
		else:
			#don't use tex or original labels
			for i, vi in self._vDict.items():
				ax.annotate('v_'+str(vi.vID), (vi.vPos[0],vi.vPos[1]), horizontalalignment='left', verticalalignment='bottom' )

		#scatter plot the graph vertices to put them on the graph
		ax.scatter(vPosMatrix[0,:], vPosMatrix[1,:], s=20, c='k', zorder=2)

		#store all the edges (with directions) that we need to draw
		arrowList = []
		#for changing the arcs of the arrows when there are multiple connections
		conStyle = "arc3,rad="
		#now let's make a list of all the arrows we want to draw
		for key, edgeList in self._eDict.items():
			vPosLeft = self._vDict[key[0]].vPos #co-ords of left vertex
			vPosRight = self._vDict[key[1]].vPos #co-ords of right vertex
			#the key is a tuple (i,j) where the left vertex has internal ID i and right vertex has internal ID j
			#edgeList is a list of all the edges between these two vertices, in this direction.
			#we need to draw as many arrows as there are elements in edgeList
			for eNumber in range(len(edgeList)):
				#draw one arrow for each entry in edgeList
				eConStyle = conStyle + str(eNumber * radFac)
				arrow_eNumber = pts.FancyArrowPatch(vPosLeft, vPosRight, connectionstyle=eConStyle,**kw)
				arrowList.append(arrow_eNumber)
		#add the arrows to the plot
		for arrow in arrowList:
			ax.add_patch(arrow)
		if show:
			#show the resulting plot
			fig.show()
		return fig, ax

	def ConstructM(self, derivToo=False):
		'''
		Constructs the Weyl-Titchmarsh M-function (matrix in this case) for the graph.
		INPUTS:
			derivToo: 	(optional) bool, if True then we will also construct and return the function M'(w). Default False
		OUTPUTS:
			MMatrix: 	lambda function, MMatrix(w,theta) is the value of the M-matrix function at the value w and for the quasimomentum theta provided
			dMMatrix: 	(optional) lambda function, dMMatrix(w,theta) is the value of the function M'(w, theta). Returned if derivToo is True.
		'''
		#we will create a dictionary containing all the connection information
		#this dictionary will have keys (j,k) for each edge j-->k in the graph
		#the value at key (j,k) will be a list of two elements
		#the first element being a numpy array of the lengths l_jk for each of the edges e_jk
		#the second element being a numpy array of the qm coefficients for each edge e_jk
		#both arrays are stacked horizontally, so that lengths and qms in each match by column index
		connectionDict = {}
		
		#gather information from _eDict
		for key, edgeList in self._eDict.items():
			nEdges = len(edgeList)
			lengths = np.zeros((nEdges,), dtype=float)
			qms = np.zeros((2, nEdges), dtype=float)
			for i, edge in enumerate(edgeList):
				lengths[i] = edge.length
				qms[:,i] = edge.qm
			#now place this information into the connectionDict
			connectionDict[key] = [lengths, qms]
		#this means that when we need to compute an entry (j,k) of M, we just lookup the keys to get the information we need and then use vector operations to sum the terms together
		
		#now we actually need to construct the M-matrix
		edgeKeys = self._eDict.keys()
		
		def MMatrix(omega, theta):
			'''
			Evaluates the M-matrix of the graph G at (omega, theta) for spectral parameter omega and quasi-momentum theta.
			INPUTS:
				omega: 	float, spectral parameter to evaluate M at
				theta: 	(2,) float numpy array, value of the quasi-momentum to evaluate M at
			OUTPUTS:
				M: 	(_nVert, _nVert) complex numpy array, the value of the M matrix at the input values
			'''
			
			M = np.zeros((self._nVert, self._nVert), dtype=complex)
			#NB j is the imaginary unit in python, so we loop over i
			for i in range(self._nVert):
				for k in range(self._nVert):
					if ((i,k) in edgeKeys) and (i != k):
						#this is a non-zero entry of the M-matrix, and i \neq k case
						#first obtain the contribution from i-->k;
						l_ik = connectionDict[i,k][0]
						qm_ik = np.sum( connectionDict[i,k][1] * theta[:, np.newaxis], 0 ) #multiply the column vector theta by the qm coefficients provided for each edge, then sum the components to get the value theta_jk on each edge
						sum_ik = omega * np.sum( np.exp( 1.j * qm_ik * l_ik ) * csc( l_ik * omega ) ) #sum contribution for j-->k
						#now if there are edges going in the opposite direction, account for those too
						if (k,i) in edgeKeys:
							#there are also edges going in the opposite direction, so we need this contribution too
							l_ki = connectionDict[k,i][0]
							qm_ki = np.sum( connectionDict[k,i][1] * theta[:, np.newaxis], 0 )
							sum_ki = omega * np.sum( np.exp( -1.j * qm_ki * l_ki ) * csc( l_ki * omega ) )
						else:
							sum_ki = 0. + 0.j
						M[i,k] = -sum_ik - sum_ki
					elif ((i,k) in edgeKeys) and (i == k):
						#this is a non-zero entry of the M-matrix, and i == k case, AND there are loops
						#first gather the contributions from loops, if there are any
						l_ii = connectionDict[i,i][0]
						qm_ii = np.sum( connectionDict[i,i][1] * theta[:, np.newaxis], 0 )
						sum_ii = 2 * omega * np.sum( cot(l_ii*omega) - cos(qm_ii*l_ii)*csc(l_ii*omega) )
						#now gather contributions from the other edges that connect to i
						sum_il = 0. + 0.j
						sum_li = 0. + 0.j
						for l in range(self._nVert):
							if i==l: #don't need this one again!
								continue
							if ((i,l) in edgeKeys):
								#i --> l is an edge, gather the contribution from this connection
								l_il = connectionDict[i,l][0]
								sum_il += omega * cot(l_il * omega)
							if ((l,i) in edgeKeys):
								#l --> i is an edge, gather the contribution from this connection
								l_li = connectionDict[l,i][0]
								sum_li += omega * cot(l_li * omega)
						#now the element M[i,k]=M[k,k]=M[i,i] can be constructed :)
						M[i,i] = sum_ii + sum_il + sum_li
					elif i==k:
						#even if there aren't any loops, this could still be a non-zero entry in the M-matrix
						sum_il = 0. + 0.j
						sum_li = 0. + 0.j
						for l in range(self._nVert):
							if i==l: #don't need this one again!
								continue
							if ((i,l) in edgeKeys):
								#i --> l is an edge, gather the contribution from this connection
								l_il = connectionDict[i,l][0]
								sum_il += omega * cot(l_il * omega)
							if ((l,i) in edgeKeys):
								#l --> i is an edge, gather the contribution from this connection
								l_li = connectionDict[l,i][0]
								sum_li += omega * cot(l_li * omega)
						M[i,i] = sum_li + sum_il
			return M
		#this completes the function that constructs the M-matrix, we can now return MMatrix as the output, unless we also require the derivative thanks to derivToo==True
		if derivToo:
			#need to build the derivative matrix (wrt omega) too!
			
			def dMMatrix(omega, theta):
				'''
				Evaluates the element-wise derivative (wrt omega) of the M-matrix of the graph G at (omega, theta) for spectral parameter omega and quasi-momentum theta.
			INPUTS:
				omega: 	float, spectral parameter to evaluate M at
				theta: 	(2,) float numpy array, value of the quasi-momentum to evaluate M at
			OUTPUTS:
				dM: 	(_nVert, _nVert) complex numpy array, the value of the element-wise derivative (wrt omega) of the M matrix at the input values
				'''
				
				dM = np.zeros((self._nVert, self._nVert), dtype=complex)

				for i in range(self._nVert):
					for k in range(self._nVert):
						if ((i,k) in edgeKeys) and (i != k):
							#this is a non-zero entry of the dM-matrix, and i \neq k case
							#first obtain the contribution from i-->k;
							l_ik = connectionDict[i,k][0]
							qm_ik = np.sum( connectionDict[i,k][1] * theta[:, np.newaxis], 0 ) #multiply the column vector theta by the qm coefficients provided for each edge, then sum the components to get the value theta_jk on each edge
							sum_ik = np.exp( 1.j * qm_ik * l_ik ) * csc(l_ik*omega) * ( omega*l_ik*cot(l_ik*omega) - 1 )
							#now if there are edges going in the opposite direction, account for those too
							if (k,i) in edgeKeys:
								#there are also edges going in the opposite direction, so we need this contribution too
								l_ki = connectionDict[k,i][0]
								qm_ki = np.sum( connectionDict[k,i][1] * theta[:, np.newaxis], 0 )
								sum_ki = np.exp( -1.j*qm_ki*l_ki ) * csc(l_ki*omega) * (omega*l_ki*cot(l_ki*omega) - 1)
							else:
								sum_ki = 0. + 0.j
							dM[i,k] = -sum_ik - sum_ki
						elif ((i,k) in edgeKeys) and (i == k):
							#this is a non-zero entry of the dM-matrix, and i == k case
							#first gather the contributions from loops, if there are any
							l_ii = connectionDict[i,i][0]
							qm_ii = np.sum( connectionDict[i,i][1] * theta[:, np.newaxis], 0 )
							sum_ii = 2 * cos(qm_ii*l_ii)*csc(l_ii*omega) * ( omega*l_ii*cot(l_ii*omega) - 1 )
							sum_ii += 2 * cot(l_ii*omega)
							sum_ii -= 2 * omega*l_ii*csc(l_ii*omega)*csc(l_ii*omega)
							#now gather contributions from the other edges that connect to i
							sum_il = 0. + 0.j
							sum_li = 0. + 0.j
							for l in range(self._nVert):
								if i==l: #don't need this one again!
									continue
								if ((i,l) in edgeKeys):
									#i --> l is an edge, gather the contribution from this connection
									l_il = connectionDict[i,l][0]
									sum_il += cot(l_il*omega)
									sum_il -= omega*l_il*csc(l_il*omega)*csc(l_il*omega)
								if ((l,i) in edgeKeys):
									#l --> i is an edge, gather the contribution from this connection
									l_li = connectionDict[l,i][0]
									sum_li += cot(l_li*omega)
									sum_li -= omega*l_li*csc(l_li*omega)*csc(l_li*omega)
							#now the element dM[i,k]=dM[k,k]=dM[i,i] can be constructed :)
							dM[i,i] = sum_ii + sum_il + sum_li
						elif i==k:
							#even if there aren't any loops, this could still be a non-zero entry in the dM-matrix
							sum_il = 0. + 0.j
							sum_li = 0. + 0.j
							for l in range(self._nVert):
								if i==l: #don't need this one again!
									continue
								if ((i,l) in edgeKeys):
									#i --> l is an edge, gather the contribution from this connection
									l_il = connectionDict[i,l][0]
									sum_il += cot(l_il*omega)
									sum_il -= omega*l_il*csc(l_il*omega)*csc(l_il*omega)
								if ((l,i) in edgeKeys):
									#l --> i is an edge, gather the contribution from this connection
									l_li = connectionDict[l,i][0]
									sum_li += cot(l_li*omega)
									sum_li -= omega*l_li*csc(l_li*omega)*csc(l_li*omega)
							dM[i,i] = sum_li + sum_il
				return dM
			#this completes the function that constructs the dM-matrix, so now we return both functions
			return MMatrix, dMMatrix
		else:
			#just return MMatrix function
			return MMatrix
		#shouldn't get to here, if we do something has gone wrong
		return
	
	def ConstructAltM(self, derivToo=True, denomToo=True):
		'''
		Constructs the M-matrix for this graph, but having pulled outside the matrix a factor of omega/prod(sin(lengths)). That is, all entries of the matrix no longer have reciprocal trig functions in them (as these are taken to the outside of the matrix) but in their place have products of sin functions.
		INPUTS:
			derivToo: 	(optional) bool - default True, if True then we will also construct and return the function AltM'(w). Default False
			denomToo: 	(optional) bool - default True, if True then we will also return a function that evaluates the prefactor omega/prod(sin(lengths)). 
		OUTPUTS:
			AltM: 	lambda function, AltM(w,theta) is the value of the alternative form of the M-matrix function at the value w and for the quasimomentum theta provided
			dAltM: 	(optional) lambda function, dAltM(w,theta) is the value of the function AltM'(w, theta). Returned if derivToo is True.
			Mdenom: 	(optional) lambda function, Mdenom(w) is the value of prod(sin(lengths))/omega. Note that 1/ this function is the prefactor that was pulled out of the M-matrix. The value at 0 is set as the limit value 0 to avoid division errors.
		'''
		#the plan is as follows:
		# - create an array of all the unique lengths of edges in the graph, uniqueLengths
		# - create a dictionary for all the connections, with keys (j,k) for edge j-->k in the graph.
		## Each entry in the connection dictionary will be a list of two elements,
		## the first being a numpy array of indices for uniqeLengths. Each index is that of the length in uniqueLengths that is to be ignored when adding the contribution of the edge e_jk to the element M[j,k]
		## the second being a numpy array of qm coefficients for each edge e_jk
		## both arrays are stacked horizontally, so that indices and qms match by column index
		uniqueLengths = []
		for key, edgeList in self._eDict.items():
			for edge in edgeList:
				uniqueLengths.append(edge.length)
		uniqueLengths = np.unique(np.asarray(uniqueLengths))
		
		connectionDict = {}
		for key, edgeList in self._eDict.items():
			nEdges = len(edgeList)
			ignoreIndex = np.zeros((nEdges,), dtype=int)
			qms = np.zeros((2,nEdges), dtype=float)
			for i, edge in enumerate(edgeList):
				qms[:,i] = edge.qm
				ignoreIndex[i] = np.where(uniqueLengths==edge.length)[0][0] #this is the index of the corresponding length in uniqueLengths
			connectionDict[key] = [ignoreIndex, qms]
		#this means that when we need to compute an entry (j,k) of M, we just lookup the keys to get the information we need about this entry
		
		#now we actually need to construct the M-matrix
		edgeKeys = self._eDict.keys()
		
		def AltM(omega, theta):
			'''
			Evaluates the result of pulling a factor of omega/prod(sin(lengths)) out of the M-matrix of the graph G at (omega, theta) for spectral parameter omega and quasi-momentum theta.
			INPUTS:
				omega: 	float, spectral parameter to evaluate M at
				theta: 	(2,) float numpy array, value of the quasi-momentum to evaluate M at
			OUTPUTS:
				M: 	(_nVert, _nVert) complex numpy array, the value of the M matrix at the input values
			'''
			
			M = np.zeros((self._nVert, self._nVert), dtype=complex)
			#NB j is the imaginary unit in python, so we loop over i
			for i in range(self._nVert):
				for k in range(self._nVert):
					if ((i,k) in edgeKeys) and (i != k):
						#this is a non-zero entry of the M-matrix, and i \neq k case
						#first obtain the contribution from i-->k
						#get the lengths
						l_ik = uniqueLengths[connectionDict[i,k][0]]
						#now construct the qms
						qm_ik = np.sum( connectionDict[i,k][1] * theta[:, np.newaxis], 0 ) #multiply the column vector theta by the qm coefficients provided for each edge, then sum the components to get the value theta_jk on each edge.
						#this makes qm_ik a 1D numpy array, with qm_ik[j] being theta_ik for the jth edge connecting i-->k
						#now we need to evaluate the product of sin(l*omega) for all lengths excluding the one for the edges!
						prods_ik = np.zeros_like(l_ik, dtype=complex)
						for j in range(np.shape(prods_ik)[0]):
							otherLengths = uniqueLengths[np.arange(len(uniqueLengths))!=connectionDict[i,k][0][j]] #array of all lengths barring the length of (the jth edge corresponding to the connection) e_ik
							prods_ik[j] = np.prod( sin( otherLengths * omega ) )#product of sin(lengths * omega) for all edge lengths barring e_ik's
						sum_ik = np.sum( np.exp( 1.j * qm_ik * l_ik ) * prods_ik ) #sum contribution for i-->k
						#now if there are edges going in the opposite direction, account for those too
						if (k,i) in edgeKeys:
							#there are also edges going in the opposite direction, so we need this contribution too
							l_ki = uniqueLengths[connectionDict[k,i][0]]
							qm_ki = np.sum( connectionDict[k,i][1] * theta[:, np.newaxis], 0 )
							prods_ki = np.zeros_like(l_ki, dtype=complex)
							for j in range(np.shape(prods_ki)[0]):
								otherLengths = uniqueLengths[np.arange(len(uniqueLengths))!=connectionDict[k,i][0][j]]
								prods_ki[j] = np.prod( sin( otherLengths * omega ) )
							sum_ki = np.sum( np.exp( -1.j * qm_ki * l_ki ) * prods_ki )
						else:
							sum_ki = 0. + 0.j
						M[i,k] = -sum_ik - sum_ki
					elif ((i,k) in edgeKeys) and (i == k):
						#this is a non-zero entry of the M-matrix, and i == k case, AND there are loops
						#first gather the contributions from loops, if there are any
						l_ii = uniqueLengths[connectionDict[i,i][0]]
						qm_ii = np.sum( connectionDict[i,i][1] * theta[:, np.newaxis], 0 )
						prods_ii = np.zeros_like(l_ii, dtype=complex)
						for j in range(np.shape(prods_ii)[0]):
							otherLengths = uniqueLengths[np.arange(len(uniqueLengths))!=connectionDict[i,i][0][j]]
							prods_ii[j] = np.prod( sin( otherLengths * omega ) )
						sum_ii = 2 * np.sum( prods_ii * cos(l_ii*omega) - cos(qm_ii*l_ii) * prods_ii )
						#now gather contributions from the other edges that connect to i
						sum_il = 0. + 0.j
						sum_li = 0. + 0.j
						for l in range(self._nVert):
							if i==l: #don't need this one again!
								continue
							if ((i,l) in edgeKeys):
								#i --> l is an edge, gather the contribution from this connection
								l_il = uniqueLengths[connectionDict[i,l][0]]
								prods_il = np.zeros_like(l_il, dtype=complex)
								for j in range(np.shape(prods_il)[0]):
									otherLengths = uniqueLengths[np.arange(len(uniqueLengths))!=connectionDict[i,l][0][j]]
									prods_il[j] = np.prod( sin( otherLengths * omega ) )
								sum_il += prods_il * cos(l_il * omega)
							if ((l,i) in edgeKeys):
								#l --> i is an edge, gather the contribution from this connection
								l_li = uniqueLengths[connectionDict[l,i][0]]
								prods_li = np.zeros_like(l_li, dtype=complex)
								for j in range(np.shape(prods_li)[0]):
									otherLengths = uniqueLengths[np.arange(len(uniqueLengths))!=connectionDict[l,i][0][j]]
									prods_li[j] = np.prod( sin( otherLengths * omega ) )
								sum_li += prods_li * cos(l_li * omega)
						#now the element M[i,k]=M[k,k]=M[i,i] can be constructed :)
						M[i,i] = sum_ii + sum_il + sum_li
					elif i==k:
						#even if there aren't any loops, this could still be a non-zero entry in the M-matrix
						sum_il = 0. + 0.j
						sum_li = 0. + 0.j
						for l in range(self._nVert):
							if i==l: #don't need this one again!
								continue
							if ((i,l) in edgeKeys):
								#i --> l is an edge, gather the contribution from this connection
								l_il = uniqueLengths[connectionDict[i,l][0]]
								prods_il = np.zeros_like(l_il, dtype=complex)
								for j in range(np.shape(prods_il)[0]):
									otherLengths = uniqueLengths[np.arange(len(uniqueLengths))!=connectionDict[i,l][0][j]]
									prods_il[j] = np.prod( sin( otherLengths * omega ) )
								sum_il += prods_il * cos(l_il * omega)
							if ((l,i) in edgeKeys):
								#l --> i is an edge, gather the contribution from this connection
								l_li = uniqueLengths[connectionDict[l,i][0]]
								prods_li = np.zeros_like(l_li, dtype=complex)
								for j in range(np.shape(prods_li)[0]):
									otherLengths = uniqueLengths[np.arange(len(uniqueLengths))!=connectionDict[l,i][0][j]]
									prods_li[j] = np.prod( sin( otherLengths * omega ) )
								sum_li += prods_li * cos(l_li * omega)
						M[i,i] = sum_li + sum_il
			return M
		#this completes the function that constructs the M-matrix, we can now return AltM as the output, unless we also require the derivative or denominator
		if derivToo:
			def dAltM(omega, theta):
				'''
				Evaluates the element-wise derivative (wrt omega) of the AltM function of the graph G at (omega, theta) for spectral parameter omega and quasi-momentum theta.
			INPUTS:
				omega: 	float, spectral parameter to evaluate dAltM at
				theta: 	(2,) float numpy array, value of the quasi-momentum to evaluate M at
			OUTPUTS:
				dM: 	(_nVert, _nVert) complex numpy array, the value of the element-wise derivative (wrt omega) of the AltM function at the input values	
				'''
				dM = np.zeros((self._nVert, self._nVert), dtype=complex)
				#NB j is the imaginary unit in python, so we loop over i
				for i in range(self._nVert):
					for k in range(self._nVert):
						if ((i,k) in edgeKeys) and (i != k):
							#this is a non-zero entry of the dM-matrix, and i \neq k case
							#first obtain the contribution from i-->k
							#get the lengths
							l_ik = uniqueLengths[connectionDict[i,k][0]]
							#now construct the qms
							qm_ik = np.sum( connectionDict[i,k][1] * theta[:, np.newaxis], 0 ) #multiply the column vector theta by the qm coefficients provided for each edge, then sum the components to get the value theta_jk on each edge.
							#this makes qm_ik a 1D numpy array, with qm_ik[j] being theta_ik for the jth edge connecting i-->k
							#now we need to evaluate the sum of l_lm cos(l_lm)*prod(sin(l*omega)) for all lengths excluding the one for the edge e_ik, which is the derivative of this term in the M-matrix wrt omega
							sumProds_ik = np.zeros_like(l_ik, dtype=complex)
							for j in range(np.shape(sumProds_ik)[0]):
								otherLengths = uniqueLengths[np.arange(len(uniqueLengths))!=connectionDict[i,k][0][j]] #array of all lengths barring the length of (the jth edge corresponding to the connection) e_ik
								nLengths = np.shape(otherLengths)[0]
								#now we have removed l_ik[j] from the lengths array, but need to evaluate the sum * product that arises from taking the derivative... the plan for this is to stack copies of otherLengths as rows of a square matrix. Then take sin() of all the off-diagonal terms and l_lm*cos() of the diagonal terms, then take a row-wise product. The product of each row is then l_lm*cos()*prod(sin()), which we can then sum to get the sum over l_lm of all these terms
								termsMatrix = np.tile(otherLengths * omega, (nLengths, 1)) #creates a square matrix with otherLengths*omega repeated along the rows
								termsMatrix[range(nLengths), range(nLengths)] = 0. #set diagonal terms to zero so can take sin() of whole matrix w/o changing the diagonal
								termsMatrix = sin(termsMatrix) #take sin of all terms - we added omega into our construction, so termsMatrix is already lengths*omega
								termsMatrix += np.diag( otherLengths * cos(otherLengths*omega) ) #add diagonal terms
								sumProds_ik[j] = np.sum( np.prod( termsMatrix, 1 ) ) #get product of the rows of termsMatrix, then sum them
							sum_ik = np.sum( np.exp( 1.j * qm_ik * l_ik ) * sumProds_ik ) #sum contribution for i-->k
							#now if there are edges going in the opposite direction, account for those too
							if (k,i) in edgeKeys:
								#there are also edges going in the opposite direction, so we need this contribution too
								l_ki = uniqueLengths[connectionDict[k,i][0]]
								qm_ki = np.sum( connectionDict[k,i][1] * theta[:, np.newaxis], 0 )
								sumProds_ki = np.zeros_like(l_ki, dtype=complex)
								for j in range(np.shape(sumProds_ki)[0]):
									otherLengths = uniqueLengths[np.arange(len(uniqueLengths))!=connectionDict[k,i][0][j]]
									nLengths = np.shape(otherLengths)[0]
									termsMatrix = np.tile(otherLengths * omega, (nLengths, 1))
									termsMatrix[range(nLengths), range(nLengths)] = 0.
									termsMatrix = sin(termsMatrix)
									termsMatrix += np.diag( otherLengths * cos(otherLengths*omega) )
									sumProds_ki[j] = np.sum( np.prod( termsMatrix, 1 ) )
								sum_ki = np.sum( np.exp( -1.j * qm_ki * l_ki ) * sumProds_ki )
							else:
								sum_ki = 0. + 0.j
							dM[i,k] = -sum_ik - sum_ki
						elif ((i,k) in edgeKeys) and (i == k):
							#this is a non-zero entry of the M-matrix, and i == k case, AND there are loops
							#first gather the contributions from loops, if there are any			
#note: formula for this term!
## \sum_{j\sim l}\cos(l_{jl}\omega)*\diff{}{\omega}\bracs{\Pi_{not j\sim l}\sin(l_{mn}\omega)}
## - \sum_{j\sim l}l_{jl} \Pi_{all m,n}\sin(l_{mn}\omega)
## + 2\sum_{j\rCon j}\cos(l_{jj}\omega)\diff{}{\omega}\bracs{\Pi_{not j\rCon j}\sin(l_{mn}\omega)}
## - 2\sum_{j\rCon j}l_{jj} \Pi_{all m,n}\sin(l_{mn}\omega)
## - 2\sum_{j\rCon j}cos(\qm_{jj}l_{jj}) \Pi_{not j\rCon j}\sin(l_{mn}\omega)					
							l_ii = uniqueLengths[connectionDict[i,i][0]]
							qm_ii = np.sum( connectionDict[i,i][1] * theta[:, np.newaxis], 0 )
							#sumProds assembles \diff{}{\omega}\bracs{\Pi_{not j\rCon j}\sin(l_{mn}\omega)}
							sumProds_ii = np.zeros_like(l_ii, dtype=complex)
							#sinProds assembles \Pi_{all m,n}\sin(l_{mn}\omega), this is the same for both loop and non-loop connection contributions
							sinProd =  np.prod( sin(uniqueLengths * omega) )
							for j in range(np.shape(sumProds_ii)[0]):
								otherLengths = uniqueLengths[np.arange(len(uniqueLengths))!=connectionDict[i,i][0][j]]
								nLengths = np.shape(otherLengths)[0]
								termsMatrix = np.tile(otherLengths * omega, (nLengths, 1))
								termsMatrix[range(nLengths), range(nLengths)] = 0.
								termsMatrix = sin(termsMatrix)
								termsMatrix += np.diag( otherLengths * cos(otherLengths*omega) )
								sumProds_ii[j] = np.sum( np.prod( termsMatrix, 1 ) )
							sum_ii = 2 * np.sum( cos(l_ii * omega) * sumProds_ii )
							sum_ii += -2 * np.sum( l_ii * sinProd )
							sum_ii += -2 * np.sum( cos(qm_ii * l_ii) * sumProds_ii )
							#now gather contributions from the other edges that connect to i
							sum_il = 0. + 0.j
							sum_li = 0. + 0.j
							for l in range(self._nVert):
								if i==l: #don't need this one again!
									continue
								if ((i,l) in edgeKeys):
									#i --> l is an edge, gather the contribution from this connection
									l_il = uniqueLengths[connectionDict[i,l][0]]
									sumProds_il = np.zeros_like(l_il, dtype=complex)
									for j in range(np.shape(sumProds_il)[0]):
										otherLengths = uniqueLengths[np.arange(len(uniqueLengths))!=connectionDict[i,l][0][j]] #array of all lengths barring the length of (the jth edge corresponding to the connection) e_ik
										nLengths = np.shape(otherLengths)[0]
										termsMatrix = np.tile(otherLengths * omega, (nLengths, 1))
										termsMatrix[range(nLengths), range(nLengths)] = 0. 
										termsMatrix = sin(termsMatrix) 
										termsMatrix += np.diag( otherLengths * cos(otherLengths*omega) )
										sumProds_il[j] = np.sum( np.prod( termsMatrix, 1 ) )
									sum_il += cos(l_il * omega) * sumProds_il
									sum_il += -1. * np.sum( l_il * sinProd )
								if ((l,i) in edgeKeys):
									#l --> i is an edge, gather the contribution from this connection
									l_li = uniqueLengths[connectionDict[l,i][0]]
									sumProds_li = np.zeros_like(l_li, dtype=complex)
									for j in range(np.shape(sumProds_li)[0]):
										otherLengths = uniqueLengths[np.arange(len(uniqueLengths))!=connectionDict[l,i][0][j]] #array of all lengths barring the length of (the jth edge corresponding to the connection) e_ik
										nLengths = np.shape(otherLengths)[0]
										termsMatrix = np.tile(otherLengths * omega, (nLengths, 1))
										termsMatrix[range(nLengths), range(nLengths)] = 0. 
										termsMatrix = sin(termsMatrix) 
										termsMatrix += np.diag( otherLengths * cos(otherLengths*omega) )
										sumProds_li[j] = np.sum( np.prod( termsMatrix, 1 ) )
									sum_li += cos(l_li * omega) * sumProds_li
									sum_li += -1. * np.sum( l_li * sinProd )
							dM[i,i] = sum_ii + sum_il + sum_li
						elif i==k:
							#even if there aren't any loops, this could still be a non-zero entry in the M-matrix
							sinProd =  np.prod( sin(uniqueLengths * omega) )
							sum_il = 0. + 0.j
							sum_li = 0. + 0.j
							for l in range(self._nVert):
								if i==l: #don't need this one again!
									continue
								if ((i,l) in edgeKeys):
									#i --> l is an edge, gather the contribution from this connection
									l_il = uniqueLengths[connectionDict[i,l][0]]
									sumProds_il = np.zeros_like(l_il, dtype=complex)
									for j in range(np.shape(sumProds_il)[0]):
										otherLengths = uniqueLengths[np.arange(len(uniqueLengths))!=connectionDict[i,l][0][j]] #array of all lengths barring the length of (the jth edge corresponding to the connection) e_ik
										nLengths = np.shape(otherLengths)[0]
										termsMatrix = np.tile(otherLengths * omega, (nLengths, 1))
										termsMatrix[range(nLengths), range(nLengths)] = 0. 
										termsMatrix = sin(termsMatrix) 
										termsMatrix += np.diag( otherLengths * cos(otherLengths*omega) )
										sumProds_il[j] = np.sum( np.prod( termsMatrix, 1 ) )
									sum_il += cos(l_il * omega) * sumProds_il
									sum_il += -1. * np.sum( l_il * sinProd )
								if ((l,i) in edgeKeys):
									#l --> i is an edge, gather the contribution from this connection
									l_li = uniqueLengths[connectionDict[l,i][0]]
									sumProds_li = np.zeros_like(l_li, dtype=complex)
									for j in range(np.shape(sumProds_li)[0]):
										otherLengths = uniqueLengths[np.arange(len(uniqueLengths))!=connectionDict[l,i][0][j]] #array of all lengths barring the length of (the jth edge corresponding to the connection) e_ik
										nLengths = np.shape(otherLengths)[0]
										termsMatrix = np.tile(otherLengths * omega, (nLengths, 1))
										termsMatrix[range(nLengths), range(nLengths)] = 0. 
										termsMatrix = sin(termsMatrix) 
										termsMatrix += np.diag( otherLengths * cos(otherLengths*omega) )
										sumProds_li[j] = np.sum( np.prod( termsMatrix, 1 ) )
									sum_li += cos(l_li * omega) * sumProds_li
									sum_li += -1. * np.sum( l_li * sinProd )
							dM[i,i] = sum_li + sum_il
				return dM
		#if wanted, construct the prefactor function too
		if denomToo:
			def Mdenom(omega):
				'''
				Evaluates 1/ the prefactor that was pulled out of the M-matrix to construct AltM.
				That is, this function returns
					product( sin(l_jk*omega) ) / omega,
				with the product taken over all unique edge lengths l_jk.
				The function is vectorised, so can handle multiple omega.
				INPUTS:
					omega: 	(n,) float numpy array, values of omega to evaluate the function at
				OUTPUTS:
					val: 	(n,) float numpy array, values of the factor as described. If an entry in omega is zero, returns the limiting value of 0.
				'''
				
				#note: if the length of uniqueLengths is only 1, then the limit at zero is the sole length in uniqueLengths, otherwise it is zero
				if np.shape(uniqueLengths)[0]==1:
					val = np.prod( np.sin( np.outer(uniqueLengths, omega) ), axis=0 ) / omega
					val[omega==0] = uniqueLengths[0]
				else:				
					val = np.prod( np.sin( np.outer(uniqueLengths, omega) ), axis=0 ) / omega
					val[omega==0] = 0. #set as limit value for omega=0
				return val
		
		#check which ones we were told to construct
		if derivToo and denomToo:
			return AltM, dAltM, Mdenom
		elif derivToo:
			return AltM, dAltM
		elif denomToo:
			return AltM, Mdenom
		#if none of the above, then we only needed AltM
		return AltM