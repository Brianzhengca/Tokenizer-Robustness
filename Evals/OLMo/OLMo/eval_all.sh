#!/bin/bash
#SBATCH --partition=gpu-a40
#SBATCH --account=ark
#SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-gpu=5
#SBATCH --gres=gpu:8
#SBATCH --time=1-20:00:00
#SBATCH --job-name=run_evals
#SBATCH --output="run_evals.out"

source ~/.bashrc
conda deactivate
conda activate open-instruct-run
export HF_TOKEN=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

model_name=mistralai/Mistral-7B-Instruct-v0.3
qa_format=qa
num_incontext_examples=0
max_num_examples=500

#tasks=("arc-easy" "arc-challenge" "arithmetic" "boolq" "code-description" "commonsenseqa" "copa" "coqa" "cs-algorithms" "cute" "drop" "dyck-languages" "gsm" "hellaswag" "hotpotqa" "humaneval" "jeopardy" "lambada" "language-identification" "lsat" "mbpp" "mmlu" "openbookqa" "operators" "piqa" "repeat-copy-logic" "squad" "tofu" "triviaqa" "wikidataqa" "winograd" "winogrande")
#tasks=("arc-easy")
#tasks=("arithmetic" "code-description" "coqa" "cs-algorithms" "cute" "drop" "dyck-languages" "gsm" "hotpotqa" "jeopardy" "lambada" "language-identification" "lsat" "mbpp" "operators" "repeat-copy-logic" "squad" "tofu" "triviaqa" "wikidataqa")
#tasks=("arc-easy" "coqa" "arc-challenge" "arithmetic" "boolq" "code-description" "commonsenseqa" "copa" "cs-algorithms" "cute" "drop" "dyck-languages" "gsm" "hellaswag" "hotpotqa" "jeopardy" "lambada" "language-identification" "lsat" "mbpp" "mmlu" "openbookqa" "operators" "piqa" "repeat-copy-logic" "squad" "tofu" "triviaqa" "wikidataqa" "winograd" "winogrande")
#tasks=("cute")
tasks=("arc-easy" "arc-challenge" "copa" "winogrande" "winograd" "commonsenseqa" "openbookqa" "piqa" "mmlu" "boolq" "hellaswag" "wikidataqa" "tofu" "triviaqa" "jeopardy" "cute" "gsm" "drop")

for task in ${tasks[@]}
do
    if [ $task == "coqa" ]; then
        output_dir=Mistral/normal_results/$task-$qa_format/$model_name/$step
    elif [ $task == "humaneval" ] || [ $task == "mbpp" ]; then
        output_dir=Mistral/normal_results/$task/$model_name/$step
    else
        output_dir=Mistral/normal_results/$task-$qa_format-ice${num_incontext_examples}/$model_name/$step
    fi

    if [ ! -d $output_dir ]; then
        id=$(sbatch --parsable --export=all,model_name=$model_name,step=$step,num_incontext_examples=$num_incontext_examples,output_dir=$output_dir,qa_format=$qa_format,max_num_examples=$max_num_examples,eval_batch_size=16 scripts/sbatch/eval/eval_${task}.sh)
        echo "  $task: Submitted batch job $id"
    fi
done