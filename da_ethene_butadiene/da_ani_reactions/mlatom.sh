#!/bin/bash
#SBATCH -J da_ani
#SBATCH -w l011
#SBATCH -p lab
#SBATCH -c 6
#SBATCH -G 1
#SBATCH -o Job%J_%x_%N_out
#SBATCH -e Job%J_%x_%N_err

hostname
# Command here
# export PGI_FASTMATH_CPU=sandybridge
export PYTHONPATH=/mlatom/home/yifanhou/git/mlatom-dev
python md.py > md.out
#
sleep 1
