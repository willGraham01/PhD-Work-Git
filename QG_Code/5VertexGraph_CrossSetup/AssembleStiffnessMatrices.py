##For the details of the task which this script performs, see Assemble_Stiffness_Matrix.ipynb.

if __name__ == "__main__":
    from fenics import *
    from dolfin import *
    import scipy.sparse as sp

    # This is the name of our mesh files, minus the extension.
    filePrefix = 'FEM_CrossGraphSetup'
    # This is the gmsh file which generated our domain's mesh. We can infer information about our domain from it
    gmshFile = filePrefix + '.geo'
    # This is the folder into which we will place our stiffness matrices, once they are assembled
    matDumpFolder = 'StiffnessMatrixDump'

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

    # Create filenames to save outputs to
    fileA1Name = './' + matDumpFolder + '/a1_N' + Nstr
    fileA2Name = './' + matDumpFolder + '/a2_N' + Nstr

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

    # Create function space for the problem,
    # making sure we tell the function space that the domain is periodic
    V = FunctionSpace(mesh, 'CG', 1, constrained_domain=PeriodicDomain())

    # Define variational problem for A_1 and A_2
    u = TrialFunction(V)
    v = TestFunction(V)
    A1 = dot(grad(u), grad(v))*dx
    A2 = u*v*dx

    # Now build the stiffness matrices A_1 and A_2.
    # Save these to files in CSR format since CSC isn't available
    # To aid with memory problems, we build one at a time and overwrite the same variable after writing to a file
    # Build A_1
    A = assemble(A1)
    A_mat = as_backend_type(A).mat()
    A_sparray = sp.csr_matrix(A_mat.getValuesCSR()[::-1], shape = A_mat.size) #store in sparse format
    sp.save_npz(fileA1Name, A_sparray)
    # Now build A_2
    A = assemble(A2)
    A_mat = as_backend_type(A).mat()
    A_sparray = sp.csr_matrix(A_mat.getValuesCSR()[::-1], shape = A_mat.size) #store in sparse format
    sp.save_npz(fileA2Name, A_sparray)
