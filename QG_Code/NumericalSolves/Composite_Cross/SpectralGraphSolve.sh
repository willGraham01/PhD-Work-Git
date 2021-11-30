#!/bin/bash

source activate PythonBasics

cd /home/will/Documents/PhD/PhD-Work-Git/QG_Code/NumericalSolves/Composite_Cross/

# TEST: FDM thinks that, with N=31 gridpoints in each direction, that omega=4.075999077339437
# is an eigenvalue with quasimomentum = 0.
# Let's see what the spectral method spits out at each stage

M=3
N=5
omega=4.075999077339437
t1=0.0
t2=0.0
fDump='./DtNEfuncs/FDM-CompatabilityCheck/'
fOut='M3-N5.npz'
resFile='./DtNEfuncs/FDM-CompatabilityCheck/resultFile.csv'
prevInfo='./DtNEfuncs/FDM-CompatabilityCheck/M3-N4.npz'

# Compute approximations to the DtN eigenfunctions,
# The script creates a "relay" file for us to use, and deletes this file later
#python DtN_Minimiser.py -M $M -N $N -omega $omega -t1 $t1 -t2 $t2 -fOut $fOut -fDump $fDump -fDump $fDump
python DtN_Minimiser.py -M $M -N $N -omega $omega -t1 $t1 -t2 $t2 -fOut $fOut -fDump $fDump -prevInfo $prevInfo -nIts 5000

infoFile=$(cat DtN-Minimiser-ConveyerFile.txt)
# Having computed the approximations to the DtN eigenfunctions, let's try checking if we have an eigenvalue?
python OnGraphSpectralSolve.py -infoFile $infoFile -k 3 -tol 1e-8 -fOut $resFile

# clean up relay file
rm ./DtN-Minimiser-ConveyerFile.txt
# clean up DtN eigenfunction data if we really want to
# rm $infoFile
