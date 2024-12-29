#!/bin/bash
#SBATCH --partition=gpu-l40s
#SBATCH --account=ark
#SBATCH --mem-per-gpu=28G
#SBATCH --cpus-per-gpu=5
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --job-name=run_codeline
#SBATCH --output="run_codeline.out"

conda init
conda activate tokenizer-robustness

python generate_codeline.py