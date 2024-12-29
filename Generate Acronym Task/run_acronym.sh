#!/bin/bash
#SBATCH --partition=gpu-l40s
#SBATCH --account=ark
#SBATCH --mem-per-gpu=50G
#SBATCH --cpus-per-gpu=2
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --job-name=run_acronym
#SBATCH --output="run_acronym.out"

conda init
conda activate tokenizer-robustness

python generate_acronym.py