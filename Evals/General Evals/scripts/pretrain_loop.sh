config=configs/alisa/OLMo2-7B-generic200k.yaml

# wandb_run_id=TTnazKRN #AQFCECqu
# export MODEL_NAME=OLMo2-7B-pt200k
# export TOKENIZER=om2-pt200k

# wandb_run_id=NTUPjYbh
# export MAX_SEQUENCE_LENGTH=$(python -c "print(round(4096*4.458679110036542/6.096777900247578))")
# export MODEL_NAME=OLMo2-7B-pts200k-t180k-ctx${MAX_SEQUENCE_LENGTH}
# export TOKENIZER=om2-pts200k-t180k
# export TRAIN_STEPS=$(python -c "print(round(76533*6.096777900247578/4.458679110036542))")

# wandb_run_id=ovfvSlwi
# export MAX_SEQUENCE_LENGTH=2752
# export TRAIN_STEPS=118595
# export TOKENIZER=om2-pts200k-t80k-mw4
# export MODEL_NAME=OLMo2-7B-pts200k-t80k-ctx${MAX_SEQUENCE_LENGTH}

# wandb_run_id=yupewWRQ
# export MAX_SEQUENCE_LENGTH=2759
# export TRAIN_STEPS=118269
# export TOKENIZER=om2-pts200k-t100k-mw4
# export MODEL_NAME=OLMo2-7B-pts200k-t100k-ctx${MAX_SEQUENCE_LENGTH}

# wandb_run_id=SGyDeRet
# export MAX_SEQUENCE_LENGTH=2818
# export TRAIN_STEPS=115584
# export TOKENIZER=om2-pts200k-t140k-mw4
# export MODEL_NAME=OLMo2-7B-pts200k-t140k-ctx${MAX_SEQUENCE_LENGTH}

# wandb_run_id=hWcuzUTV
# export MAX_SEQUENCE_LENGTH=2881
# export TRAIN_STEPS=112840
# export TOKENIZER=om2-pts200k-t160k-mw4
# export MODEL_NAME=OLMo2-7B-pts200k-t160k-ctx${MAX_SEQUENCE_LENGTH}

# wandb_run_id=nEdBisVW
# export MAX_SEQUENCE_LENGTH=2756
# export TRAIN_STEPS=118409
# export TOKENIZER=om2-pts200k-t80k-mw4-colon
# export MODEL_NAME=OLMo2-7B-pts200k-t80k-ctx${MAX_SEQUENCE_LENGTH}-colon

# wandb_run_id=UUgqEftb
# export MAX_SEQUENCE_LENGTH=3000
# export TRAIN_STEPS=107972
# export TOKENIZER=om2-pts200k-t180k-mw4-colon
# export MODEL_NAME=OLMo2-7B-pts200k-t180k-ctx${MAX_SEQUENCE_LENGTH}-colon

# wandb_run_id=UniiDEQI
# export MAX_SEQUENCE_LENGTH=2884
# export TRAIN_STEPS=112712
# export TOKENIZER=om2-pts200k-t160k-mw4-colon
# export MODEL_NAME=OLMo2-7B-pts200k-t160k-ctx${MAX_SEQUENCE_LENGTH}-colon

wandb_run_id=FmYICncT
export MAX_SEQUENCE_LENGTH=3000
export TRAIN_STEPS=107972
export TOKENIZER=om2-pts200k-t180k-mw4-colon
export MODEL_NAME=OLMo2-7B-pts200k-t180k-ctx${MAX_SEQUENCE_LENGTH}-colon

echo "wandb_run_id: $wandb_run_id"
echo "config: $config"
echo "MAX_SEQUENCE_LENGTH: $MAX_SEQUENCE_LENGTH"
echo "TRAIN_STEPS: $TRAIN_STEPS"
echo "MODEL_NAME: $MODEL_NAME"
echo "TOKENIZER: $TOKENIZER"

# id=$(sbatch --parsable --export=all,config=$config,wandb_run_id=$wandb_run_id scripts/sbatch/pretrain.sh)
# echo "Submitted job with id $id"

id=2146619
while true
do
    id=$(sbatch --parsable --export=all,config=$config,wandb_run_id=$wandb_run_id --dependency=afterany:$id scripts/sbatch/resume_pretrain.sh)
    echo "Submitted job with id $id"
done
