from batchedGeneration import Generator
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompts import PROMPTS

import pickle as pkl

import torch
import random
import numpy as np
import sys

sys.setrecursionlimit(50000)

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

model_name = "meta-llama/Llama-3.1-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
model.generation_config.pad_token_id = tokenizer.eos_token_id
prompt = PROMPTS["del_char"]
input_prompts = []
groundtruths = []
outputfile = open("output.txt", "w")

def remove_all(string, char):
    return string.replace(char, "")

with open("valid_tokens.pkl", "rb") as f:
    tokens = pkl.load(f)
    valid_tokens = [pair[0] for pair in tokens]
    token_lengths = [pair[1] for pair in tokens]

for token in valid_tokens:
    character = random.choice(list(token))
    groundtruths.append(remove_all(token, character))
    input_prompts.append(prompt.format(character, token))

generator = Generator(input_prompts, tokenizer, model, batch_size=10)
commoncorrect = 0
commoncount = 0
rarecorrect = 0
rarecount = 0

print(input_prompts)

for index, output in enumerate(generator.generate()):
    groundtruth = groundtruths[index]
    output = output.split("<|end_header_id|>\n\n")[-1].split("<|eot_id|>")[0]
    output = output.splitlines()[-1]
    parsed_output = output.split()[-1].replace(".", "").replace('"', "").rstrip()
    print(groundtruth, parsed_output, token_lengths[index])
    outputfile.write(str(parsed_output) + " " + str(token_lengths[index]) + "\n")
    if (token_lengths[index] == 1):
        rarecount += 1
        if (groundtruth == parsed_output):
            rarecorrect += 1
    else:
        commoncount += 1
        if (groundtruth == parsed_output):
            commoncorrect += 1
print("Rarely Segmented:", rarecorrect / rarecount)
print("Commonly Segmented:", commoncorrect / commoncount)
print(rarecount)
print(commoncount)
outputfile.close()