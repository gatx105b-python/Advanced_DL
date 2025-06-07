#!/usr/bin/bash

#SBATCH -J test-run
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v11
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out

source /data/dy0718/anaconda3/bin/activate
python train.py

exit 0

