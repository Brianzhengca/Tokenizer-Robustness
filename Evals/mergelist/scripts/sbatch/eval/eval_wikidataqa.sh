#!/bin/bash
#SBATCH --partition=gpu-l40s
#SBATCH --account=cse
#SBATCH --mem-per-gpu=10G
#SBATCH --cpus-per-gpu=1
#SBATCH --gres=gpu:8
#SBATCH --time=23:59:00
#SBATCH --job-name=run_evals
#SBATCH --output="run_wikidataqa.out"

source ~/.bashrc
conda deactivate
conda activate open-instruct-run
export HF_TOKEN=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

cat $0
echo "--------------------"
date

echo "Evaluating $model_name at step $step on WikidataQA"
python -m eval.wikidataqa.run_eval \
    --model_name_or_path $model_name \
    --output_dir $output_dir \
    --num_incontext_examples $num_incontext_examples \
    ${max_num_examples:+--max_num_examples "$max_num_examples"} \
    ${eval_batch_size:+--eval_batch_size "$eval_batch_size"} \
    ${step:+--step "$step"} \
    ${qa_format:+--qa_format "$qa_format"}