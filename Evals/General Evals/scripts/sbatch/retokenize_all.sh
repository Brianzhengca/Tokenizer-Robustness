#!/bin/bash
#SBATCH --account=nvr_lacr_llm
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=nvr_lacr_llm-retokenize.om2
#SBATCH --output="slurm/retokenize/slurm-%J-%x.out"

cat $0
echo "--------------------"

date
tokenizer=om2-pts200k-t180k-mw4
echo "tokenizer: $tokenizer"
python retokenize.py olmo-mix-v2/olmo2_train_subset olmo-mix-v2/${tokenizer} tokenizers/${tokenizer}/tokenizer.json
