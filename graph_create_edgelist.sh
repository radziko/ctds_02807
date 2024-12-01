#!/bin/bash
#BSUB -J GraphBuilder
#BSUB -q hpc
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "select[model == XeonGold6226R]"
#BSUB -B
#BSUB -N
##BSUB -u s204071@dtu.dk
#BSUB -o Output_%J.out
#BSUB -e Output_%J.err
#BSUB -W 01:00 
#BSUB -n 16
#BSUB -R "span[hosts=1]"

source .venv/bin/activate

python graph_create_edgelist.py