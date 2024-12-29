from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import torch
import random
import numpy as np
import sys

sys.setrecursionlimit(50000)

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct', trust_remote_code=True, token="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")
model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-3.1-8B-Instruct',
    trust_remote_code=True,
    token="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"
)
model.to("cuda")

input_ids_prefix = tokenizer.encode(
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
    add_special_tokens=False
)
input_ids_suffix = tokenizer.encode("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)

def generate_right_tokenization(number):
    tokenization = []
    curr = []
    for i in range(len(number)-1, -1, -1):
        if (len(curr) == 3): #replace with desired chunk size
            tokenization.insert(0, ''.join(curr))
            curr = []
        curr.insert(0, number[i])
    tokenization.insert(0, ''.join(curr))
    return tokenization

prompts = [line.rstrip() for line in open("arithmetic_prompts.txt", "r").readlines()]
groundtruth = [line.rstrip() for line in open("arithmetic_groundtruth.txt", "r").readlines()]
outfile = open("r2l_inputtokens.txt", "w")
normal_correct = 0
perturbed_correct = 0

for index, prompt in enumerate(prompts):
    promptsplitted = prompt.rstrip().split()
    answer = float(groundtruth[index])
    if '-' in prompt:
        inputtokens = generate_right_tokenization(promptsplitted[0]) + ['Ġ-', 'Ġ'] + generate_right_tokenization(promptsplitted[2]) + ['Ġ=']
    elif '/' in prompt:
        inputtokens = generate_right_tokenization(promptsplitted[0]) + ['Ġ/', 'Ġ'] + generate_right_tokenization(promptsplitted[2]) + ['Ġ=']
    elif '*' in prompt:
        inputtokens = generate_right_tokenization(promptsplitted[0]) + ['Ġ*', 'Ġ'] + generate_right_tokenization(promptsplitted[2]) + ['Ġ=']
    else:
        inputtokens = generate_right_tokenization(promptsplitted[0]) + ['Ġ+', 'Ġ'] + generate_right_tokenization(promptsplitted[2]) + ['Ġ=']
    outfile.write(str(inputtokens) + "\n")
    print(tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt, add_special_tokens=False)))
    print(inputtokens)
    normal_output = model.generate(input_ids=torch.tensor(input_ids_prefix + tokenizer.encode(prompt, add_special_tokens=False) + input_ids_suffix, dtype=torch.long).unsqueeze(0).to('cuda'), max_new_tokens=512, do_sample=False)
    normal_generation = tokenizer.batch_decode(normal_output)[0]
    perturbed_output = model.generate(input_ids=torch.tensor(input_ids_prefix + tokenizer.convert_tokens_to_ids(inputtokens) + input_ids_suffix, dtype=torch.long).unsqueeze(0).to('cuda'), max_new_tokens=512, do_sample=False)
    perturbed_generation = tokenizer.batch_decode(perturbed_output)[0]
    #print(tokenizer.convert_ids_to_tokens(perturbed_output.tolist()[0]))
    print(perturbed_generation)
    normal_generation = re.sub(r"(\d),(\d)", r"\1\2", normal_generation)
    perturbed_generation = re.sub(r"(\d),(\d)", r"\1\2", perturbed_generation)
    normalNumber = float(re.findall(r"[-+]?\d*\.\d+|\d+", normal_generation)[-1])
    perturbedNumber = float(re.findall(r"[-+]?\d*\.\d+|\d+", perturbed_generation)[-1])
    print(normalNumber, perturbedNumber, answer, index)
    print("---------------------")
    try:
        if (abs(round(perturbedNumber, 5)) == answer):
            perturbed_correct += 1
    except OverflowError: # randomly generates infinity
        pass
    try:
        if (abs(round(normalNumber, 5)) == answer):
            normal_correct += 1
    except OverflowError:
        pass
outfile.close()
print("Normal: " + str(normal_correct / len(groundtruth)))
print("Perturbed: " + str(perturbed_correct / len(groundtruth)))