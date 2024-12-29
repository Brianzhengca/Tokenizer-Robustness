from batchedGeneration import Generator
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompts import PROMPTS

import pickle as pkl

model_name = "meta-llama/Llama-3.1-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
model.generation_config.pad_token_id = tokenizer.eos_token_id
prompt = PROMPTS["spell_inverse"]
input_prompts = []
outputfile = open("output.txt", "w")

with open("valid_tokens.pkl", "rb") as f:
    tokens = pkl.load(f)
    valid_tokens = [pair[0] for pair in tokens]
    token_lengths = [pair[1] for pair in tokens]

for token in valid_tokens:
    input_prompts.append(prompt.format(' '.join(list(token))))

generator = Generator(input_prompts, tokenizer, model, batch_size=10)
commoncorrect = 0
commoncount = 0
rarecorrect = 0
rarecount = 0

for index, output in enumerate(generator.generate()):
    groundtruth = valid_tokens[index]
    #print(output)
    output = output.split("<|end_header_id|>\n\n")[-1].split("<|eot_id|>")[0]
    parsed_output = output.split()[-1].replace(".", "").replace('"', "").rstrip()
    print(groundtruth, parsed_output)
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