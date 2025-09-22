#!/bin/bash
#SBATCH --account=nvr_lacr_llm
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=nvr_lacr_llm-pretrain.olmo2-7b
#SBATCH --output="slurm/train/slurm-%J-%x.out"

cat $0
echo "--------------------"
date

echo "wandb_run_id: $wandb_run_id"
echo "config: $config"
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" $(which hostname) --ip-address)
echo nodes: ${nodes}
echo head node: ${head_node} at ${head_node_ip}
srun --export=all torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=gpu \
    --start_method=fork \
    --rdzv_id=$RANDOM \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${head_node_ip}:29500 \
    scripts/train.py $config \
        --wandb.run_id=$wandb_run_id
