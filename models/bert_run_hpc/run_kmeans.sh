#!/bin/bash
#BSUB -J RunKMeansOnBertEmb
#BSUB -q hpc
#BSUB -R "rusage[mem=4GB]"
#BSUB -B
#BSUB -N
##BSUB -u s204161@dtu.dk
#BSUB -o Output_%J.out
#BSUB -e Output_%J.err
#BSUB -W 12:00 
#BSUB -n 32
#BSUB -R "span[hosts=1]"

source /dtu/blackhole/07/155527/camproj/env_works/bin/activate
python run_kmeans.py