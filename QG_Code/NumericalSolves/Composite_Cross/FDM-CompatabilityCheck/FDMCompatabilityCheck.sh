#!/bin/bash

# TEST: FDM thinks that, with N=31 gridpoints in each direction, that omega=4.075999077339437 is an eigenvalue with quasimomentum = 0,
# for the cross-in-plane geometry.
# Let's see what the spectral method spits out at each stage

source activate PythonBasics

cd /home/will/Documents/PhD/PhD-Work-Git/QG_Code/NumericalSolves/Composite_Cross/

# Ranges for M and N; will loop over these in sequence
# Safe (Ie will converge) ranges: 1-M-7, 1-N-20
Mstart=11
Mend=11
Nstart=1
Nend=15
# estimated runtimes PER eigenfunction for various M values:
# M <= 5 - less than 1/2 a minute per function
# M around 7 - about 2-3 mins
# M around 9-10 - about 10 mins per function

# parameters that should remain constant throughout all runs
# supposed eigenvalue
omega=4.075999077339437
# quasi-momentum values
t1=0.0
t2=0.0
# max. number of iterations for minimiser solve
nIts=5000
# output file locations
fDump='./FDM-CompatabilityCheck/eFuncDump/'
resFile='./FDM-CompatabilityCheck/resultFile.csv'

echo "Beginning loops, printing progress below."
for m in $(seq $Mstart $Mend); do
    for n in $(seq $Nstart $Nend); do
        echo " "
        echo "     <<< START M=$m, N=$n >>>"
        echo " "
        # setup output file name
        fOut="M$m-N$n.npz"

        # lookup previous information, provided it's available
        if [[ "$m" -eq 1 ]]; then
            # when m=1, we only have a previous run if n > 1
            if [[ "$n" -eq 1 ]]; then
                # if additionally we have n==1, we just have to wing the solve without previous information
                python DtN_Minimiser.py -M $m -N $n -omega $omega -t1 $t1 -t2 $t2 -fOut $fOut -fDump $fDump -fDump $fDump -nIts $nIts #-lOff
            else
                # n>1, so we can use the previous information in Mm-N(n-1).npz
                prevInfo="${fDump}M$m-N$(($n-1)).npz"
                python DtN_Minimiser.py -M $m -N $n -omega $omega -t1 $t1 -t2 $t2 -fOut $fOut -fDump $fDump -prevInfo $prevInfo -nIts $nIts #-lOff
            fi #end if, lol
        else
            # when m>1, we might have previous information from Mm-N(n-1), unless n==1 too, in which case we can use M(m-1)-N1
            if [[ "$n" -eq 1 ]]; then
                # use previous M, first e'function approximation instead
                prevInfo="${fDump}M$(($m-1))-N1.npz"
                python DtN_Minimiser.py -M $m -N $n -omega $omega -t1 $t1 -t2 $t2 -fOut $fOut -fDump $fDump -prevInfo $prevInfo -nIts $nIts #-lOff
            else
                # use current M, n-1 approximations
                prevInfo="${fDump}M$m-N$(($n-1)).npz"
                python DtN_Minimiser.py -M $m -N $n -omega $omega -t1 $t1 -t2 $t2 -fOut $fOut -fDump $fDump -prevInfo $prevInfo -nIts $nIts #-lOff
            fi
        fi
        # The script DtN_Minimiser creates a "relay" file for us to use, which we will delete later
        # Also, we can check that convergence occurred, otherwise there's no point in these next commands
        exitStatus=$?

        # if the exit status was 0, it might be that our prevInfo was causing some issues (this is observed for M2-N9, where using the M2-N8 data causes varphi_7 not to be computed... even though it's already converged!)

        if [[ "$exitStatus" -eq 0 ]]; then
            # DtN_Minimiser script exited with success, so proceed with e'val solve
            infoFile=$(cat DtN-Minimiser-ConveyerFile.txt)
            # Having computed the approximations to the DtN eigenfunctions, let's try checking if we have an eigenvalue?
            python OnGraphSpectralSolve.py -infoFile $infoFile -k 3 -tol 1e-8 -fOut $resFile

            # clean up relay file
            rm ./DtN-Minimiser-ConveyerFile.txt
            # clean up DtN eigenfunction data if we really want to
            # rm $infoFile
        else
            # DtN_Minimiser didn't converge, don't bother e'val solving
            echo "DtN_Minimiser exited with 1: failed convergence."
            echo "Not recording this data, break N loop"
            break
        fi
    done
done
