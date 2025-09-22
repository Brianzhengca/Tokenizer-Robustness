#!/bin/bash
#SBATCH --partition=ckpt
#SBATCH --account=ark
#SBATCH --mem-per-gpu=10G
#SBATCH --constraint="[l40|a40|l40s]"
#SBATCH --cpus-per-gpu=1
#SBATCH --gres=gpu:3
#SBATCH --time=23:59:00
#SBATCH --job-name=run_evals
#SBATCH --output="run_cbpb.out"

source ~/.bashrc
conda deactivate
conda activate open-instruct-run

cat $0
echo "--------------------"
date

model_name_or_path=models/hf_models/$model_name/step$step
batch_size=2

if [ -z "$max_context_length" ]; then
    echo "max_context_length will be the model's max sequence length"
    output_dir=results/cbpb/$model_name/$step
else
    echo "max_context_length is set to $max_context_length"
    output_dir=results/cbpb-ctx${max_context_length}/$model_name/$step
fi

python -m eval.eval_corrected_bpb \
    --model_name_or_path $model_name \
    --start_idx $start_idx \
    --num_examples $num_examples \
    --output_dir $output_dir \
    ${max_context_length:+--max_context_length $max_context_length} \
    ${batch_size:+--batch_size $batch_size}
