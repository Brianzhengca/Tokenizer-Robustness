conda init
conda activate tokenizer-robustness

python generate_mmlu.py
python evaluate_mmlu.py
