from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

import pickle as pkl

with open("instance_dict.pkl", "rb") as f:
    instance_dict = pkl.load(f) # {token : # of ways it was segmented}

print("Filtering Tokens...")

valid_tokens = []

for key, value in tqdm(instance_dict.items()):
    occurrence = sum(1 for _ in open(f"../../llama_superstring_encodings/{key}.jsonl"))
    if ((value == 1 and 5 <= len(key) <= 15) or (value > 5 and 5 <= len(key) <= 15)) and 50 < occurrence < 500:
        valid_tokens.append([key, value])

valid_tokens.sort(key = lambda x:x[1])
for index, token in enumerate(valid_tokens):
    print(token)

with open("valid_tokens.pkl", "wb") as f:
    pkl.dump(valid_tokens, f)