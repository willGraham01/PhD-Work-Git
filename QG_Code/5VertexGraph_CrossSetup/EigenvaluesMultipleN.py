# Python script to compute eigenvalues for multiple values of N in a single run.
# See documentation file Compute_For_Multiple_N.ipynb for more information.

import subprocess
import sys

if __name__=="__main__":
    # This is the name of our mesh files, minus the extension.
    filePrefix = 'FEM_CrossGraphSetup'
    # This is the gmsh file which generated our domain's mesh. We can infer information about our domain from it
    gmshFile = filePrefix + '.geo'

    # For each given input argument, interpret it as an int if possible (error if otherwise),
    # then edit the value of N as defined in gmshFile, line 12 (index 11 when file read in).
    # After doing this, go through the process of computing the eigenvalues (and saving them) as per the shell script
    # FEM_CrossGraphSolve.sh
    for arg in sys.argv:
        # first, we attempt to interpret the input arguments as ints
        try:
            N = int(arg)
            Nstr = str(arg)
            print('Working on N=%d' % (N))
        except ValueError:
            # this probably means that a float was passed in as one value of N.
            # print this to the user, but then continue on to the other input values.
            print('%s is invalid for conversion to int' % (arg))
            continue

        # we now know the value of N to edit into the mesh file, so write it
        with open(gmshFile, 'r') as file:
            # read a list of lines into data
            data = file.readlines()
        # now change the 12th line where the definition of N appears
        # note the newline character needs to be appended here
        data[11] = 'N = ' + Nstr + ';\n'
        # and finally write everything back to the file
        with open(gmshFile, 'w') as file:
            file.writelines(data)

        # Having now edited the value of N in the gmsh file, we can call the shell script to compute the eigenvalues
        # NOTE: When this file is moved to the documentation folder, this path may need to change
        subprocess.call(['sh', './FEM_CrossGraphSolve.sh'])
    print('Done')
