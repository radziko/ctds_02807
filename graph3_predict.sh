#!/bin/bash
#BSUB -J GraphProcess
#BSUB -q hpc
#BSUB -R "rusage[mem=92GB]"
#BSUB -B
#BSUB -N
##BSUB -u s204071@dtu.dk
#BSUB -o Output_%J.out
#BSUB -e Output_%J.err
#BSUB -W 24:00 
#BSUB -n 1
#BSUB -R "span[hosts=1]"

source .venv/bin/activate

python graph3_predict.py