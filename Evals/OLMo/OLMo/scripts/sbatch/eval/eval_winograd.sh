#!/bin/bash
#SBATCH --partition=gpu-a40
#SBATCH --account=ark
#SBATCH --mem-per-gpu=50G
#SBATCH --cpus-per-gpu=5
#SBATCH --gres=gpu:2
#SBATCH --time=23:00:00
#SBATCH --job-name=run_evals
#SBATCH --output="run_winograd.out"

source ~/.bashrc
conda deactivate
conda activate open-instruct-run

cat $0
echo "--------------------"
date

echo "Evaluating $model_name at step $step on Winograd"
python -m eval.winograd.run_eval \
    --model_name_or_path $model_name \
    --output_dir $output_dir \
    --num_incontext_examples $num_incontext_examples \
    ${max_num_examples:+--max_num_examples "$max_num_examples"} \
    ${eval_batch_size:+--eval_batch_size "$eval_batch_size"} \
    ${step:+--step "$step"} \
    ${qa_format:+--qa_format "$qa_format"}