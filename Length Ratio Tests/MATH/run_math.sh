#!/bin/bash
#SBATCH --partition=gpu-l40s
#SBATCH --account=ark
#SBATCH --mem-per-gpu=50G
#SBATCH --cpus-per-gpu=5
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --job-name=run_math_sampled
#SBATCH --output="run_math_sampled.out"

conda init
conda activate tokenizer-robustness

python generate_math.py