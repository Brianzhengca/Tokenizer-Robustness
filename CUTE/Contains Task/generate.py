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
prompt = PROMPTS["contains_char"]
input_prompts = []
groundtruths = []
outputfile = open("output.txt", "w")

with open("valid_tokens.pkl", "rb") as f:
    tokens = pkl.load(f)
    valid_tokens = [pair[0] for pair in tokens]
    token_lengths = [pair[1] for pair in tokens]

for token in valid_tokens:
    character = random.choice(list("qwertyuiopasdfghjklzxcvbnm"))
    if character in token:
        groundtruths.append("Yes")
    else:
        groundtruths.append("No")
    input_prompts.append(prompt.format(character, token))

generator = Generator(input_prompts, tokenizer, model, batch_size=10)
commoncorrect = 0
commoncount = 0
rarecorrect = 0
rarecount = 0

print(input_prompts)

for index, output in enumerate(generator.generate()):
    groundtruth = groundtruths[index]
    #print(output)
    output = output.split("<|end_header_id|>\n\n")[-1].split("<|eot_id|>")[0]
    output = output.splitlines()[-1]
    parsed_output = output.split()[-1].replace(".", "").replace('"', "").rstrip()
    if (parsed_output != "Yes" and parsed_output != "No"):
        if "not" in output or "no" in output:
            parsed_output = "No"
        else:
            parsed_output = "Yes"
    print(groundtruth, parsed_output, input_prompts[index])
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