#!/bin/bash
#SBATCH --partition=ckpt
#SBATCH --account=ark
#SBATCH --mem-per-gpu=10G
#SBATCH --constraint="[l40|a40|l40s]"
#SBATCH --cpus-per-gpu=1
#SBATCH --gres=gpu:3
#SBATCH --time=23:59:00
#SBATCH --job-name=run_evals
#SBATCH --output="run_openbookqa.out"

source ~/.bashrc
conda deactivate
conda activate open-instruct-run
export HF_TOKEN=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

cat $0
echo "--------------------"
date

echo "Evaluating $model_name at step $step on OpenbookQA"
python -m eval.openbookqa.run_eval \
    --model_name_or_path $model_name \
    --output_dir $output_dir \
    --num_incontext_examples $num_incontext_examples \
    ${max_num_examples:+--max_num_examples "$max_num_examples"} \
    ${eval_batch_size:+--eval_batch_size "$eval_batch_size"} \
    ${step:+--step "$step"} \
    ${qa_format:+--qa_format "$qa_format"}