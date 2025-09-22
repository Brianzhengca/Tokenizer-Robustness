#!/bin/bash
#SBATCH --account=nvr_lacr_llm
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --job-name=nvr_lacr_llm-eval.all
#SBATCH --output="slurm/eval/slurm-%J-%x.out"

cat $0
echo "--------------------"
date

echo "Evaluating $model_name at step $step on all tasks"
python -m eval.eval_all \
    --model_name_or_path models/hf_models/$model_name \
    ${step:+--step "$step"} \
    ${overwrite:+--overwrite}
