from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from batchedGeneration import TransformersInterface

import torch
import pickle as pkl
import random
import numpy as np

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

eval_set = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", trust_remote_code=True)["eval"]
# Initialize the tokenizer and model
model_name = 'meta-llama/Llama-3.1-8B-Instruct'  # Ensure this is the correct model name
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token="your hf token")
# Ensure the tokenizer has a pad token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

pad_token_id = tokenizer.pad_token_id

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, token="your hf token")

# Extract vocabulary from tokenizer
vocabulary = set(tokenizer.get_vocab().keys())

lines = []
for line in eval_set:
    lines.append(line["instruction"])
interface = TransformersInterface(model, vocabulary, tokenizer)
normal_output_file = open("normal_alpaca_outputs.txt", "w")
perturbed_output_file = open("perturbed_alpaca_outputs.txt", "w")
normal_responses, perturbed_responses = interface.generate_response(lines, max_new_tokens=512, batch_size=10)
for i in range(len(normal_responses)):
    print(normal_responses[i])
    normal_output_file.write(str([normal_responses[i]]) + "\n")
    perturbed_output_file.write(str([perturbed_responses[i]]) + "\n")
    print("Processed:", i)
normal_output_file.close()
perturbed_output_file.close()
