#!/bin/bash

# Script loops through values of theta2 to record eigenfunctions and eigenvalues for the Cross in plane geometry, via solution to the variational problem.

source activate PythonBasics

cd /home/will/Documents/PhD/PhD-Work-Git/QG_Code/NumericalSolves/Composite_Cross/

# parameters to keep constant throughout
nPts=25
nIts=750
tDim=0
N=1
M=15
# in case you want to take several slices at once
tStart=0
#tEnd=$tStart
tEnd=$(($nPts-1))
# master file name
masterName="./CompMes_VP_Results/nPts${nPts}-N${N}-t1loops${tStart}-${tEnd}.csv"
masterFName="./CompMes_VP_Results/nPts${nPts}-N${N}-t1loops${tStart}-${tEnd}-funcs.csv"

# Compute the eigenvalues and eigenfunctions, and save to files
echo "Starting 1D loops now."
for t in $(seq $tStart $tEnd); do
    fname="TEMP-nPts${nPts}-t$((${tDim}+1))loop${t}.csv"
    python CompMes_VarProbSolve.py -fn $fname -nPts $nPts -M $M -N 1 -funcs -nIts $nIts -oneD -tDim $tDim -tFixed $t
    echo "Saved to file: $fname"
done

# Merge all created csv files into a single master file
echo "Creating master file."
# less than 10 files requires only one ? to match output file names
if [[ "${tEnd}" -lt 9 ]]; then
    cat ./CompMes_VP_Results/TEMP-nPts${nPts}-t$((${tDim}+1))loop?.csv > $masterName
    cat ./CompMes_VP_Results/TEMP-nPts${nPts}-t$((${tDim}+1))loop?-funcs.csv > $masterFName
# between 10 and 99 files requires ? and ?? to match all output file names
else
    cat ./CompMes_VP_Results/TEMP-nPts${nPts}-t$((${tDim}+1))loop?.csv ./CompMes_VP_Results/TEMP-nPts${nPts}-t$((${tDim}+1))loop??.csv > $masterName
    cat ./CompMes_VP_Results/TEMP-nPts${nPts}-t$((${tDim}+1))loop?-funcs.csv ./CompMes_VP_Results/TEMP-nPts${nPts}-t$((${tDim}+1))loop??-funcs.csv > $masterFName
fi

echo "Cleaning up temporary files"
# cleanup those temporary files that you made
rm ./CompMes_VP_Results/TEMP-nPts${nPts}-t$((${tDim}+1))loop*.csv

# complete script, print reminder to append date to run information
echo "Complete: append date to the following output files:"
echo "[eVals] to ${masterName} "
echo "[eFuns] to ${masterFName} "
