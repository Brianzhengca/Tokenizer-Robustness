#!/bin/bash
#SBATCH --partition=gpu-a40
#SBATCH --account=ark
#SBATCH --mem-per-gpu=50G
#SBATCH --cpus-per-gpu=5
#SBATCH --gres=gpu:2
#SBATCH --time=23:00:00
#SBATCH --job-name=run_evals
#SBATCH --output="run_coqa.out"

source ~/.bashrc
conda deactivate
conda activate open-instruct-run
export HF_TOKEN=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

cat $0
echo "--------------------"
date

echo "Evaluating $model_name at step $step on COQA"
python -m eval.coqa.run_eval \
    --model_name_or_path $model_name \
    --output_dir $output_dir \
    --max_num_examples 300 \
    ${eval_batch_size:+--eval_batch_size "$eval_batch_size"} \
    ${step:+--step "$step"} \
    ${qa_format:+--qa_format "$qa_format"}
