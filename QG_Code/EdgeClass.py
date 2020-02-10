#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 14:12:00 2020

@author: will

File containing the Edge class that is required for solving spectral Quantum Graph problems using the M-matrix.
Dependencies:
	VertexClass.py
"""

import numpy as np
from numpy.linalg import norm

from VertexClass import Vertex

class Edge:
	'''
	Initialise with E = Edge(leftV, rightV, length='auto', qm='auto')
	An Edge object is assigned a left vertex, right vertex, length and two quasi-momentum coefficients.
	Edges are assumed to be directed from the left vertex to the right vertex.
	ATTRIBUTES:
		leftV: 	Vertex object, the left vertex for this edge
		rightV: 	Vertex object, the right vertex for this edge
		length: 	float, the length assigned to this edge
		qm: 	(2,) float numpy array, coefficients for each component of the quasi-momentum of this edge
	METHODS:
	'''
	
	def __init__(self, leftV, rightV, length='auto', qm='auto'):
		'''
		Initialisation method for Vertex.
		INPUTS:
			leftV, rightV: 	Vertex objects, the left and right vertices of this edge, respectively
			length: 	(optional) float, the length this edge is assigned. If not provided, Euclidean distance is assumed.
			qm: 	(optional) (2,) float numpy array, the coefficients for each component of the quasi-momentum of this edge. If not provided, the rotation matrix for this edge is computed and the quasi-momentum coefficients are read off from this.
		'''
		
		self.leftV = Vertex(leftV.vID, leftV.vPos, leftV.coupConst)
		self.rightV = Vertex(rightV.vID, rightV.vPos, rightV.coupConst)
		
		if isinstance(length, str):
			#no length provided, so compute it manually
			self.length = norm(rightV.vPos - leftV.vPos)
			if self.length == 0:
				raise ZeroDivisionError('Automatic length requested but computed as zero. Reminder: loops require manual length input.')
		else:
			#length provided, ignore distances
			self.length = length
		
		if isinstance(qm, str):
			#no quasi-momentum coefficients provided, compute them manually
			eI = rightV.vPos - leftV.vPos #unit vector along edge
			RI = np.asarray([ [eI[1], -eI[0]], [eI[0], eI[1]] ]) / norm(rightV.vPos - leftV.vPos) #rotation matrix R_I
			self.qm = RI[1,:]
		else:
			#coefficients provided, ignore computations
			self.qm = qm
		
		return
	
	def __repr__(self):
		'''
		Default output when an Edge object is returned to the command line
		'''
		return "Edge object connecting vertex %d --> %d" % (self.leftV.vID, self.rightV.vID)
		
	def __str__(self):
		'''
		Print text that is displayed when called with the print() function.
		'''
		retStr = "Edge object: \n Connecting vIDs %d --> %d [positions (%.2f, %.2f) --> (%.2f, %.2f)], \n length: %.2f, \n QM: (%.2f, %.2f)"		
		retTuple = (self.leftV.vID, self.rightV.vID, self.leftV.vPos[0], self.leftV.vPos[1], \
			  self.rightV.vPos[0], self.rightV.vPos[1], self.length, self.qm[0], self.qm[1])		
		return retStr % retTuple
	
#end of Edge class definition