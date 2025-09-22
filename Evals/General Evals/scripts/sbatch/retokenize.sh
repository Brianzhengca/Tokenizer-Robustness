#!/bin/bash
#SBATCH --account=nvr_lacr_llm
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=nvr_lacr_llm-retokenize.dolmino
#SBATCH --output="slurm/retokenize/slurm-%J-%x.out"

cat $0
echo "--------------------"

python retokenize.py olmo-mix-v2/${filename} olmo-mix-v2/${tokenizer}/${filename} tokenizers/${tokenizer}/tokenizer.json
