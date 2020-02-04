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

from VertexClass import Vertex
from EdgeClass import Edge

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
	METHODS:
		
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

		#now we know the number of vertices we can construct our directed adjacency matrix
		self._adjMat = np.zeros((self._nVert, self._nVert), dtype=int)
			
		#place into internal vertex dictionary, with translated ids
		for v in V:
			internalID = self._idTranslation[v.vID]
			self._vDict[internalID] = Vertex(internalID, v.vPos, v.coupConst)

		#contstruct internal edge dictionary, with translated ids
		for e in E:
			#translated vertex IDs for this edge
			transLeftID = self._idTranslation[e.leftV.vID]
			transRightID = self._idTranslation[e.rightV.vID]
			
			if (transLeftID, transRightID) in self._eDict:
				#if this key already exists in the dictionary, we are adding another edge between two vertices that already have one, so we need to append to the already created list at this key
				self.eDict[transLeftID, transRightID].append( Edge(self._vDict[transLeftID], self._vDict[transRightID], \
			  e.length, e.qm) )
			else:
				#these two edges were previously unconnected, so create a new key and list for them and add this connection
				self._eDict[transLeftID, transRightID] = [ Edge(self._vDict[transLeftID],  \
				self._vDict[transRightID], e.length, e.qm) ]
			#make a record of this edge in the adjacency matrix
			self._adjMat[transLeftID, transRightID] += 1
		
		return
	
	def adjMat(self, binary=False, undirected=False):
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
			aMat[i,j] = np.copy(self._adjMat[i,j])
		
		return aMat