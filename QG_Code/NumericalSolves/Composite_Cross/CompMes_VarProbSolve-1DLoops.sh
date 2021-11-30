#!/bin/bash

# Script loops through values of theta2 to record eigenfunctions and eigenvalues for the Cross in plane geometry, via solution to the variational problem.

source activate PythonBasics

cd /home/will/Documents/PhD/PhD-Work-Git/QG_Code/NumericalSolves/Composite_Cross/

# parameters to keep constant throughout
nPts=5
nIts=750
tDim=0
N=1
# in case you want to take several slices at once
tStart=0
tEnd=$(($nPts-1))
# master file name
masterName="./nPts${nPts}-N${N}-t1loops${tStart}-${tEnd}.csv"
masterFName="./nPts${nPts}-N${N}-t1loops${tStart}-${tEnd}-funcs.csv"

# Compute the eigenvalues and eigenfunctions, and save to files
echo "Starting 1D loops now."
for t in $(seq $tStart $tEnd); do
    fname="TEMP-nPts${nPts}-t$((${tDim}+1))loop${t}.csv"
    python CompMes_VarProbSolve.py -fn $fname -nPts $nPts -N 1 -funcs -nIts $nIts -oneD -tDim $tDim -tFixed $t
    echo "Saved to file: $fname"
done

# Merge all created csv files into a single master file
echo "Creating master file"
# less than 10 files requires only one ? to match output file names
if [[ "$tEnd" -leq 9 ]]; then
    cat "./TEMP-nPts${nPts}-t$((${tDim}+1))loop?.csv" > masterName
    cat "./TEMP-nPts${nPts}-t$((${tDim}+1))loop?-funcs.csv" > masterFName
# between 10 and 99 files requires ? and ?? to match all output file names
else
    cat "./TEMP-nPts${nPts}-t$((${tDim}+1))loop?.csv" "./TEMP-nPts${nPts}-t$((${tDim}+1))loop??.csv"  > masterName
    cat "./TEMP-nPts${nPts}-t$((${tDim}+1))loop?-funcs.csv" "./TEMP-nPts${nPts}-t$((${tDim}+1))loop??-funcs.csv" > masterFName
fi

echo "Cleaning up temporary files"
# cleanup those temporary files that you made
rm "./TEMP-nPts${nPts}-t$((${tDim}+1))loop*.csv"

# complete script, print reminder to append date to run information
echo "Complete: append date to the following output files:"
echo "[eVals] @ $masterName"
echo "[eFuns] @ $masterFName"
