tokenizer_name=om2-pts200k-t180k-mw4-colon
chunk_size=100
source_dir=olmo2_train_subset_disjoint
num_files=$(ls olmo-mix-v2/${source_dir} | wc -l)

echo "tokenizer: $tokenizer_name"

for i in $(seq 0 $chunk_size $num_files)
do
    id=$(sbatch --parsable --export=ALL,tokenizer=$tokenizer_name,start_idx=$i,chunk_size=$chunk_size,source_dir=$source_dir scripts/sbatch/retokenize_chunk.sh)
    echo "$i: Submitted job with id $id"
done
