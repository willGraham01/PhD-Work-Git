##For the details of the task which this script performs, see Get_Eigenvalues.ipynb.

if __name__=="__main__":
    from fenics import *
    from dolfin import *
    import matplotlib.pyplot as plt
    import scipy.sparse as spSparse
    import scipy.sparse.linalg as spla
    import numpy as np

    import sys

    # read the proportion of eigenvalues that we want to compute from the value passed from the command line
    if len(sys.argv) > 1:
        # we recieved an additional input argument, which we interpret as the proportion of eigenvalues that we want to compute
        if type(sys.argv[1])==float:
            evalFrac = sys.argv[1]
        else:
            try:
                evalFrac = float(sys.argv[1])
            except:
                raise TypeError('Could not interpret input "evalFrac" as float')
        print('evalFrac interpreted as %.2e' % (evalFrac))
    else:
        # no optional input provided, use default
        evalFrac = 0.1
        print('evalFrac set to default %.2e' % (evalFrac))

    # This is the name of our mesh files, minus the extension.
    filePrefix = 'FEM_CrossGraphSetup'
    # This is the gmsh file which generated our domain's mesh. We can infer information about our domain from it
    gmshFile = filePrefix + '.geo'
    # This is the folder into which we will place our stiffness matrices, once they are assembled
    matDumpFolder = 'StiffnessMatrixDump'
    evalDumpFolder = 'EvalDump'

    # Deduce value of N from gmshFile - it appears on line 12,
    # with the 5th character being (the start of) the value of N.
    with open('FEM_CrossGraphSetup.geo') as fp:
        for i, line in enumerate(fp):
            if i == 11:
                # 11th line
                Nstr = line[4:-2] #take all characters after the numerical value appears in the file
            elif i > 11:
                #terminate loop early
                break
    # We want N as a float because we will use it in computations, to define points on meshes etc. Int might cause bugs.
    for i, char in enumerate(Nstr):
        if char==';':
            Nstr = Nstr[:i]
            break
    N = float(Nstr)

    # Infer filenames via naming convention
    fA1 = './' + matDumpFolder + '/a1_N' + Nstr + '.npz'
    fA2 = './' + matDumpFolder + '/a2_N' + Nstr + '.npz'
    # Create filename to save evals to
    evalFile = './' + evalDumpFolder + './evals-N' + Nstr

    # create a subclass for the 2D-periodic domain...
    class PeriodicDomain(SubDomain):
        #map left--> right and bottom-->top

        def inside(self, x, on_boundary):
            # return True if on left or bottom boundary AND NOT on one of the two corners
            return bool((near(x[0], 0) or near(x[1], 0)) and
                    (not ((near(x[0], 0) and near(x[1], N)) or
                            (near(x[0], N) and near(x[1], 0)))) and on_boundary)

        def map(self, x, y):
            if near(x[0], N) and near(x[1], N):
                y[0] = x[0] - N
                y[1] = x[1] - N
            elif near(x[0], N):
                y[0] = x[0] - N
                y[1] = x[1]
            else:   # near(x[1], N)
                y[0] = x[0]
                y[1] = x[1] - N
    #concludes the SubDomain subclass definition

    # Now we import our mesh...
    meshFile = filePrefix + '.xml'
    mesh = Mesh(meshFile)

    # Read matrices we want in CSR format
    print('Loading stiffness matrices.')
    A1 = spSparse.load_npz(fA1)
    A2 = spSparse.load_npz(fA2)

    # Check that matrices are of the same shape, and are square
    if A1.shape != A2.shape:
        raise ValueError('Stiffness matrices have different shapes')
    elif A1.shape[0] != A1.shape[1]:
        raise ValueError('Stiffness matrices are non-square')
    else:
        nNodes = A1.shape[0]
    # the size of the stiffness matrices corresponds to the number of nodes in our mesh,
    # and hence the maximum number of eigenvalues that we can find.

    # We now need to find the eigenvalues at the bottom of the spectrum.
    # To do so, we use Scipy.sparse.linalg's generalised eigenvalue solver,
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html
    # We can do this since A1 and A2 should both be symmetric for our problem (we have an elliptic operator).
    # They might even be better behaved, but I don't think we need that here.

    # First, we can never find *all* the eigenvalues, and it would be super inefficient to try to do that too.
    # So let's just try to compute a fraction of them.
    evalsToCompute = int(np.ceil(nNodes * evalFrac))
    # We want to compute at most evalFrac (as a percentage) eigenvalues out of nNodes
    print('Computing %d eigenvalues closest to 0' % (evalsToCompute))

    # Solves A1 * x = lambda * A2 * x for an eigenpair (lambda, x).
    # Eigenvectors are stored column-wise, so x[:,i] is the eigenvector of lambda[i].
    # Also, we use lambda = omega^2 here.
    lambdaVals, wVecs = spla.eigsh(A1, k = evalsToCompute, M=A2, sigma = 0.0, which='LM', return_eigenvectors = True)

    # Check if any "eigenvalues" that were computed were below 0, and deal with them accordingly
    nNegEvals = len(lambdaVals[lambdaVals<0])
    if nNegEvals>0:
        print('Found %d negative eigenvalues, the largest of which is: %.5e - these are being removed.' \
              % (nNegEvals, np.min(lambdaVals[lambdaVals<0])) )
    else:
        print('No negative eigenvalues found.')

    # For safety, we should really discard these "negative" eigenvalues
    wVecs = wVecs[:, lambdaVals>=0]
    lambdaVals = lambdaVals[lambdaVals>=0]

    # Now we can save the legitimate eigenvalues (taking the square-root too to obtain omegas)
    # And can also save their eigenvectors
    evalSaveStr = './' + evalDumpFolder + '/evals-N' + Nstr + '-' + str(int(evalsToCompute - nNegEvals))
    evecSaveStr = './' + evalDumpFolder + '/evecs-N' + Nstr + '-' + str(int(evalsToCompute - nNegEvals))

    wVals = np.sqrt(lambdaVals) #since we've removed the <0 values, sqrt is safe
    # Save (omega, u) pairs to output folder
    np.save(evalSaveStr, wVals)
    np.save(evecSaveStr, wVecs)
    print('Saved to output files %s (evals) and %s (evecs)' % (evalSaveStr, evecSaveStr))
