#!/bin/bash
#BSUB -J GraphBuilder
#BSUB -q hpc
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "select[model == EPYC9554]"
#BSUB -B
#BSUB -N
##BSUB -u s204071@dtu.dk
#BSUB -o Output_%J.out
#BSUB -e Output_%J.err
#BSUB -W 08:00 
#BSUB -n 16
#BSUB -R "span[hosts=1]"

source .venv/bin/activate

python graph2.py