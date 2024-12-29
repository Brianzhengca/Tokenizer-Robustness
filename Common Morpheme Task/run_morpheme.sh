#!/bin/bash
#SBATCH --partition=gpu-l40
#SBATCH --account=ark
#SBATCH --mem-per-gpu=50G
#SBATCH --cpus-per-gpu=5
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --job-name=run_morpheme
#SBATCH --output="run_morpheme.out"

conda init
conda activate tokenizer-robustness

python generate_morpheme.py