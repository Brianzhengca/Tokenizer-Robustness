# model_name=OLMo2-7B-pt200k
# model_name=OLMo2-7B-pts200k-t100k-ctx2759
# model_name=OLMo2-7B-pts200k-t140k-ctx2818
# model_name=OLMo2-7B-pts200k-t160k-ctx2881
# model_name=OLMo2-7B-pts200k-t180k-ctx2995
model_name=OLMo2-7B-pts200k-t80k-ctx2756-colon
# model_name=OLMo2-7B-pts200k-t160k-ctx2884-colon
# model_name=OLMo2-7B-pts200k-t180k-ctx3000-colon

# other eval params
qa_format=qa
num_incontext_examples=5
max_num_examples=1000

echo "Model: $model_name"
echo "QA format: $qa_format"

# complete list of tasks
tasks=("arc-easy" "arc-challenge" "arithmetic" "boolq" "code-description" "commonsenseqa" "copa" "coqa" "cs-algorithms" "cute" "drop" "dyck-languages" "hellaswag" "hotpotqa" "humaneval" "jeopardy" "lambada" "language-identification" "lsat" "mmlu" "openbookqa" "operators" "piqa" "repeat-copy-logic" "squad" "tofu" "triviaqa" "wikidataqa" "winograd" "winogrande")
# tasks=("humaneval")

# collect steps in a list
steps=()
for dir in $(ls models/hf_models/$model_name)
do
    step=$(echo $dir | sed 's/step//')
    if [ $((step % 5000)) -eq 0 ]; then
        steps+=($step)
    fi
    # steps+=($step)
done

# sort steps as integers in reverse order
steps=($(echo ${steps[@]} | tr ' ' '\n' | sort -nr | tr '\n' ' '))
# take only first element (last step)
# steps=(${steps[0]})

# loop over steps
for step in ${steps[@]}
do
    echo "Step $step"
    for task in ${tasks[@]}
    do
        if [ $task == "coqa" ]; then
            output_dir=results/$task-$qa_format/$model_name/$step
        elif [ $task == "humaneval" ]; then
            output_dir=results/$task/$model_name/$step
        else
            output_dir=results/$task-$qa_format-ice${num_incontext_examples}/$model_name/$step
        fi

        if [ ! -d $output_dir ]; then
            id=$(sbatch --parsable --export=all,model_name=$model_name,step=$step,num_incontext_examples=$num_incontext_examples,output_dir=$output_dir,qa_format=$qa_format,max_num_examples=$max_num_examples scripts/sbatch/eval/eval_${task}.sh)
            echo "  $task: Submitted batch job $id"
        fi
    done
done
