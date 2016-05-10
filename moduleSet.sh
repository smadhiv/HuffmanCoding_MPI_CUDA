#! /bin/bash 
var="$1"
if [ "$var" = "run" ]
then
	module purge
	module load gcc/4.4 cuda-toolkit/5.5.22 mpich2/1.4-eth
echo "run set done"
elif [ "$var" = "compile" ] 
then
	module purge
	module load gcc/4.4 cuda-toolkit/5.5.22 mpich2/1.4
echo "compile done"
fi
