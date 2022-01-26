#!/bin/bash

# Script loops through values of theta2 to record eigenfunctions and eigenvalues for the Cross in plane geometry, via solution to the Finite Difference approximation.

source activate PythonBasics

cd /home/will/Documents/PhD/PhD-Work-Git/QG_Code/NumericalSolves/Composite_Cross/

# parameters to keep constant throughout
nPts=51
tDim=0
# check sparse matrices in use when setting this parameter!
N=301
nEvals=10
# in case you want to take several slices at once
tStart=0
#tEnd=$tStart
tEnd=$(($nPts-1))
# master file name
masterName="./FDM_Results/nPts${nPts}-N${N}-t1loops${tStart}-${tEnd}.csv"
masterFName="./FDM_Results/nPts${nPts}-N${N}-t1loops${tStart}-${tEnd}-funcs.csv"
# in case you want to set the value of alpha3?
#a=1.0
a=0.0

# Compute the eigenvalues and eigenfunctions, and save to files
echo "Starting 1D loops now."
for t in $(seq $tStart $tEnd); do
    fname="TEMP-nPts${nPts}-t$((${tDim}+1))loop${t}.csv"
    python CompMes_FDMSolve.py -fn $fname -nPts $nPts -N $N -nEvals $nEvals -a $a -funcs -oneD -tDim $tDim -tFixed $t
    echo "Saved to file: $fname"
done

# Merge all created csv files into a single master file
echo "Creating master file."
# less than 10 files requires only one ? to match output file names
if [[ "${tEnd}" -lt 9 ]]; then
    cat ./FDM_Results/TEMP-nPts${nPts}-t$((${tDim}+1))loop?.csv > ./FDM_Results/TEMP-M.csv
    cat ./FDM_Results/TEMP-nPts${nPts}-t$((${tDim}+1))loop?-funcs.csv > ./FDM_Results.TEMP-F.csv
# between 10 and 99 files requires ? and ?? to match all output file names
else
    cat ./FDM_Results/TEMP-nPts${nPts}-t$((${tDim}+1))loop?.csv ./FDM_Results/TEMP-nPts${nPts}-t$((${tDim}+1))loop??.csv > ./FDM_Results/TEMP-M.csv
    cat ./FDM_Results/TEMP-nPts${nPts}-t$((${tDim}+1))loop?-funcs.csv ./FDM_Results/TEMP-nPts${nPts}-t$((${tDim}+1))loop??-funcs.csv > ./FDM_Results/TEMP-F.csv
fi

# remove bracers so that numpy can reinterpret it's own inputs...
sed 's/[()]//g' ./FDM_Results/TEMP-M.csv > $masterName
sed 's/[()]//g' ./FDM_Results/TEMP-F.csv > $masterFName

echo "Cleaning up temporary files"
# cleanup those temporary files that you made
rm ./FDM_Results/TEMP-nPts${nPts}-t$((${tDim}+1))loop*.csv
rm ./FDM_Results/TEMP-M.csv ./FDM_Results/TEMP-F.csv

# complete script, print reminder to append date to run information
echo "Complete: append date to the following output files:"
echo "[eVals] to ${masterName} "
echo "[eFuns] to ${masterFName} "
