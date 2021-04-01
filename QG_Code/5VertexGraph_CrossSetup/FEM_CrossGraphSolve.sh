# This script will create a mesh from the file ./FEM_CrossGraphSetup.msh,
# assemble the FEM stiffness matrices,
# and compute (a proportion of the total number of) eigenvalues and eigenvectors
# of the operator, saving the outputs to files at each stage.

# See the Documentation folder for details.

source activate fenicsproject

cd /home/will/Documents/PhD/PhD-Work-Git/QG_Code/5VertexGraph_CrossSetup/

# Create the gmsh mesh from the current FEM_Domain_v2.geo file
# For some reason trying to create the gmsh file in the command line throws up an error (which I don't understand). So for the time being we will need to compile via the GUI
gmsh ./FEM_CrossGraphSetup.geo -2 -o ./FEM_CrossGraphSetup.msh
dolfin-convert ./FEM_CrossGraphSetup.msh ./FEM_CrossGraphSetup.xml
# clean-up auxillary .msh file from gmsh
rm ./FEM_CrossGraphSetup.msh

# Using this mesh, create the matrices associated with the FEM problem and save them, via FEniCs and Python
python AssembleStiffnessMatrices.py

# Having assembled the stiffness matrices, now compute the eigenvalues and eigenvectors.
# The additional argument specifies the fraction of the total number of eigenvalues to compute.
python ComputeEigenvalues.py 0.2
