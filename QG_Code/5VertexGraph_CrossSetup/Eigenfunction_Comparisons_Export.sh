#!/bin/bash

# Script quickly renders the file Eigenfunction_Comparisons.ipynb with nbfancy,
# so it can be quickly exported and sent to others to read.

# See the Documentation folder for details.

source activate JupyterKernels

cd /home/will/Documents/PhD/PhD-Work-Git/QG_Code/5VertexGraph_CrossSetup/

# First, render the file.
# Note that nbfancy takes a directory as an input, then attempts to render
# all notebooks in there, so be careful!
nbfancy render ./

# Create the folders that nbfancy expects to exist
mkdir code
mkdir data

# Now convert to html
nbfancy html

# Move output to same directory as original notebook
mv ./html/Eigenfunction_Comparisons.html ./

#Cleanup the extra folders we had to make
# Note: will ask for confirmation for the first two folders in case of name similarities!
rm -ri code
rm -ri data
# These folders are only created as a by-product of nbfancy, so can safely be deleted.
rm -r html
rm -r nbfancy
