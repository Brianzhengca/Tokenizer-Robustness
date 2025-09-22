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
python retokenize.py olmo-mix-v2/${source_dir} olmo-mix-v2/${tokenizer}-disjoint tokenizers/${tokenizer}/tokenizer.json $start_idx $chunk_size
