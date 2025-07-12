#!/bin/bash
#SBATCH --job-name=dna_train
#SBATCH --output=dna_train.out
#SBATCH --error=dna_train.err
#SBATCH --partition=gpu              # or use 'cpu' if GPU not available
#SBATCH --gres=gpu:1                 # request 1 GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00              # Max runtime (hh:mm:ss)

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate base

# Move to the directory with the code
cd ~/equivariant_electron_density/ml-dna

# Run the training script
python run_train_dna.py
