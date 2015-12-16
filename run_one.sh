#/bin/bash

FILESIZE=$1
WORKER=$2
MAXITER=$3
LAMBDA=$4
#echo $FILESIZE

name=${FILESIZE}_${WORKER}_${MAXITER}_${LAMBDA}
#echo $name
screen -dmS $name /bin/csh -c "spark-submit --driver-memory 16G pglasso.py $WORKER $MAXITER $LAMBDA data/S_$FILESIZE.csv Output/S_output_${name}.csv > Output/log_${name}"
