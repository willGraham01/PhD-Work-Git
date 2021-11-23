#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 09:00:23 2021

@author: will

This file allows the content of OnGraph_FEMSolve.ipynb to be called as a script from the command line.
For documentation, see the aforementioned file.

REQUIRES FILE:
	DtN_Minimiser.py (for FourierFunction import)
"""

#%% Imports

# sys.argv[1:] contains command line arguments. Index 0 is just the script name
import argparse

import sys

import numpy as np
from numpy import pi

from scipy.sparse.linalg import eigs

from DtN_Minimiser import FourierFunction

#%% Matrix construction functions

def BuildB(M, theta, returnDiag=True):
    '''
    Constructs the matrix B, defined above, for the value of M that we are using.
    INPUTS:
        M: int, highest order of Fourier modes being used to approximate solution
        theta: (2,) float, value of the quasi-momentum
        returnDiag: bool, if True then B is returned as a (2(2M+1),) float of the diagonal entries
    OUTPUTS:
        B: (2(2M+1),2(2M+1)) float, the (diagonal) matrix B above.
    '''

    # stack two copies of -M, -M+1,...,M-1,M on top of each other, then multiply by 2pi
    # this gives the 2\pi\alpha terms
    B = 2.*pi * np.resize(np.arange(-M,M+1,dtype=complex),(2,(2*M+1)))
    # add theta1 to the top row and theta2 to the bottom row, giving the 2\pi\alpha + theta_i terms
    B += theta.reshape((2,1))
    # square all entries
    B *= B

    if returnDiag:
        # reshape into a column vector and return
        return B.reshape((2*(2*M+1),))
    else:
        # return a diagonal matrix
        return np.diag( B.reshape((2*(2*M+1),)) )

def BuildC(M, returnDiag=True):
    '''
    Constructs the matrix C, defined above.
    INPUTS:
        M: int, highest order of Fourier modes being used to approximate solution
        returnDiag: bool, if True then C is returned as a (2(2M+1),) float of the diagonal entries
    OUTPUTS:
        C: (2(2M+1),2(2M+1)) float, the (diagonal) matrix C above.
    '''

    if returnDiag:
        return np.ones((2*(2*M+1),), dtype=complex)
    else:
        return np.eye(2*(2*M+1), dtype=complex)

def CoeffMatrix(M, phiFunctions):
    '''
    Constructs the matrix mathcalC above, given the number of Fourier modes we're using, and the
    N DtN e'functions varphi_n as FourierFunction objects.
    INPUTS:
        M: int, highest order Fourier modes we are using to approximate
        phiFunctions: list of FourierFunctions, containing the varphi_n in order
    OUTPUTS:
        mathcalC: (2(2M+1),N) complex, the array defined above.
    '''

    N = len(phiFunctions)
    mathcalC = np.zeros((2*(2*M+1),N), dtype=complex)

    for n, phi in enumerate(phiFunctions):
        # sum along the beta axis of cMat, giving the slice mathcalC[:2M+1,n]
        mathcalC[:2*M+1,n] = np.sum(phiFunctions[n].cMat, axis=1)
        # sum along the alpha axis of cMat, giving the slice mathcalC[2M+1:,n]
        mathcalC[2*M+1:,n] = np.sum(phiFunctions[n].cMat, axis=0)
    # prefactor of 2
    mathcalC *= 2.
    # sums should involve conjugates of phi terms
    mathcalC = np.conjugate(mathcalC)
    return mathcalC

def BuildL(M, phiFunctions):
    '''
    Constructs the matrix L above, given the varphi as FourierFunctions and the highest order Fourier mode
    INPUTS:
        M: int, highest order Fourier modes we are using to approximate
        phiFunctions: list of FourierFunctions, containing the varphi_n in order
    OUTPUTS:
        L: (2(2M+1),2(2M+1)) complex, the array defined above.
    '''

    # populate the array of eigenvalues from the functions passed in
    N = len(phiFunctions)
    lbdaArray = np.zeros((N,), dtype=float)
    for n in range(N):
        lbdaArray[n] = phiFunctions[n].lbda

    # prepare to construct L... for loop because I'm a terrible person and this is some 4d tensor stuff
    L = np.zeros((2*(2*M+1),2*(2*M+1)),dtype=complex)
    mathcalC = CoeffMatrix(M, phiFunctions)
    for n in range(2*(2*M+1)):
        for m in range(2*(2*M+1)):
            # build L[n,m]
            L[n,m] = np.sum( mathcalC[m,:] * np.conjugate(mathcalC[n,:]) * lbdaArray )

    return L

def BuildM(M, omega, theta, phiFunctions):
    '''
    Constructs the matrix M from B,C and L.
    INPUTS:
        M: int, highest order Fourier modes we are using to approximate
        omega: float, value of omega
        theta: (2,) float, value of the quasimomentum
        phiFunctions: list of FourierFunctions, containing the varphi_n in order
    OUTPUTS:
        matM: (2(2M+1),2(2M+1)) complex, the array defined above.
    '''

    # slightly faster to add two vectors then turn into diagonal matrix than to return diagonal matrices
    matM = np.diag( BuildB(M, theta, returnDiag=True) - omega*omega*BuildC(M, returnDiag=True) )
    # now add the contribution from L
    matM += BuildL(M, phiFunctions)
    # return the matrix that has been assembled
    return matM

#%% Command-line script execution

if __name__=='__main__':

	parser = argparse.ArgumentParser(description='Determines whether the value omega is an eigenvalue, for the Cross-in-plane geometry. Assembles "spectral method" matrix and checks for a zero eigenvalue.')
	parser.add_argument('-infoFile', type=str, help='<Required> Path to file containing DtN eigenfunction information, output of DtN-Minimiser.py')
	parser.add_argument('-k', default=6, type=int, help='Number of eigenvalues closest to zero for the solver to retrieve')
	parser.add_argument('-tol', default=1e-8, type=float, help='Tolerance to which an eigenvalue is accepted as being zero')
	parser.add_argument('-fOut', default='', type=str, help='Output file to write record of M, N, and the value of the eigenvalue of matM cloest to 0.')

	args = parser.parse_args()

	infoFile = args.infoFile
	prevInfo = np.load(infoFile) # has attributes, M, N, omega, theta, cVecs

	omega = prevInfo['omega']
	theta = prevInfo['theta']
	M = prevInfo['M']
	N = prevInfo['N']

	phiList = []
	for n in range(N):
	    phiList.append(FourierFunction(theta, omega, prevInfo['cVecs'][n,:]))

	matM = BuildM(M, omega, theta, phiList)

	# if matM has a zero eigenvalue, then omega^2 is an eigenvalue with eigenfunction reconstructable from the e'vector
	eVals, eVecs = eigs(matM, k=args.k, sigma=0.)
	# first, find the e'value closest (in absolute value) to 0
	smallInd = np.argmin(np.abs(eVals))
	print('mMat e-val closest to 0: ', eVals[smallInd])

	# if a filename was provided, save M, N, real(smallest e'val of matM), imag(smallest e'val)
	# to the file specified
	if args.fOut:
		# now write to the file
		with open(args.fOut, 'a') as f:
			f.write('%d,%d,%f,%f,\n' % (M, N, np.real(eVals[smallInd]), np.imag(eVals[smallInd])))

	if np.any(np.abs(eVals) <= args.tol):
		print('matM has a zero eigenvalue for omega = %f + %f i' % (np.real(omega), np.imag(omega)))
		sys.exit(0)
	else:
		print('No zero eigenvalue found for omega = %f + %f i' % (np.real(omega), np.imag(omega)))
		sys.exit(1)
