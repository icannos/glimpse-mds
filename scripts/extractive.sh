#!/bin/bash

#SBATCH --partition=main                     # Ask for unkillable job
#SBATCH --gres=gpu:1
#SBATCH --mem=10G                                        # Ask for 10 GB of RAM
#SBATCH --time=2:00:00                                   # The job will run for 3 hours
#SBATCH --output=./logs/abstractive_out.txt
#SBATCH --error=./logs/abstractive_error.txt
#SBATCH -c 2


# 1. Load the required modules
module --quiet load miniconda/3
module load cuda12
conda activate "glimpse"

python glimpse/data_loading/generate_abstractive_candidates.py
#python glimpse/src/compute_rsa.py --summaries data/candidates/extractive_sentences-_-2017-_-none-_-2024-10-28-14-47-50.csv
