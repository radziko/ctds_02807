#!/bin/bash
#BSUB -J GraphBuilder
#BSUB -q hpc
#BSUB -R "rusage[mem=2GB]"
#BSUB -B
#BSUB -N
##BSUB -u s204071@dtu.dk
#BSUB -o Output_%J.out
#BSUB -e Output_%J.err
#BSUB -W 08:00 
#BSUB -n 64
#BSUB -R "span[hosts=1]"

source .venv/bin/activate

python graph.py