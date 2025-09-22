model_name=OLMo2-7B-pt200k
echo "Converting model $model_name to HF"

for step in $(ls models/$model_name | grep step)
do
    step=$(echo $step | grep -o '[0-9]\+')
    if [ $step -eq 0 ]; then
        continue
    fi
    input_dir=models/$model_name/step$step
    output_dir=models/hf_models/$model_name/step$step
    if [ ! -d $output_dir ]; then
        id=$(sbatch --parsable --export=ALL,input_dir=$input_dir,output_dir=$output_dir scripts/sbatch/convert_olmo2_to_hf.sh)
        echo "$step: Submitted job with id $id"
    fi
done
