#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 14:12:00 2020

@author: will

File containing the Vertex class that is required for solving spectral Quantum Graph problems using the M-matrix.
"""

class Vertex:
	'''
	Initialise with V = Vertex(vID, vPos, coupConst=0)
	A Vertex object is assigned an ID, position, and coupling constant.
	This class is the bottom of the pile for a Graph, providing a basis for Edges.
	ATTRIBUTES:
		vID: 	int, vertex ID number
		vPos: 	(2,) numpy array, [x,y]-co-ordinate of the vertex
		coupConst: 	float, value of the coupling constant at this vertex 
	METHODS:
	'''
	
	def __init__(self, vID, vPos, coupConst=0):
		'''
		Initialisation method for Vertex.
		Coupling constant by default is set to 0, if not provided.
		'''
		
		self.vID = vID
		self.vPos = vPos
		self.coupConst = coupConst
		
		return
	
	def __repr__(self):
		'''
		Default output when a Vertex object is returned to the command line
		'''
		return "Vertex object with ID %d" % (self.vID)
		
	def __str__(self):
		'''
		Print text that is displayed when called with the print() function.
		'''
		
		return "Vertex object: \n vID: %d, \n vPos: (%.2f,%.2f) \n alpha: %.5f" % (self.vID, self.vPos[0], self.vPos[1], self.coupConst)

#end of Vertex class definition