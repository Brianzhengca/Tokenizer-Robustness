import os
import pandas as pd
from tqdm import tqdm
import pickle as pkl
import base64

encoding_path = "superstring_encodings"
instance_dict = dict()
dirs = os.listdir(encoding_path)

def decode_filename(encoded_filename):
    # Decode from URL-safe Base64
    decoded_bytes = base64.urlsafe_b64decode(encoded_filename)
    # Convert bytes back to string
    original_token = decoded_bytes.decode("utf-8")
    return original_token

def get_relevant_tokens(lst, target):
    start, end = 0, len(lst)
    for i in range(len(lst)):
        combined = ""
        j = i
        # Combine elements until the combined string's length matches or exceeds the target
        while j < len(lst):
            combined += lst[j]
            j += 1
            # If the target string is found in the combined string
            if target in combined:
                if j - i < end - start:
                    start = i
                    end = j
    relevant_tokens = lst[start:end]
    joined = "".join(relevant_tokens)
    start_idx = joined.find(target)
    extra_left = start_idx
    extra_right = len(joined) - (start_idx + len(target))
    i = 0
    while i < len(relevant_tokens) and extra_left > 0:
        if len(relevant_tokens[i]) <= extra_left:
            # This entire element is extraneous on the left side
            extra_left -= len(relevant_tokens[i])
            relevant_tokens[i] = ""  # Mark for removal later
        else:
            # Remove only part of this element
            relevant_tokens[i] = relevant_tokens[i][extra_left:]
            extra_left = 0
        i += 1
    i = len(relevant_tokens) - 1
    while i >= 0 and extra_right > 0:
        if len(relevant_tokens[i]) <= extra_right:
            # This entire element is extraneous on the right side
            extra_right -= len(relevant_tokens[i])
            relevant_tokens[i] = ""
        else:
            # Remove only part of this element
            relevant_tokens[i] = relevant_tokens[i][:-extra_right]
            extra_right = 0
        i -= 1
    return [elem for elem in relevant_tokens if elem]
for index, filepath in enumerate(dirs):
    print("Processing:", index, "of", len(dirs))
    token_of_interest = filepath.replace(".jsonl", "")
    fullpath = encoding_path + "/" + filepath
    df = pd.read_json(fullpath, lines=True)
    segmentation_set = set() # stores the different possible segmentations of a single token
    for i, row in tqdm(df.iterrows(), total=len(df)):
        if len(row["encoded_tokens"]) > 1000:
            continue
        relevant_tokens = tuple(get_relevant_tokens(row["encoded_tokens"], token_of_interest))
        segmentation_set.add(relevant_tokens)
    if (len(segmentation_set) == 0 and len(df) > 0):
        print(token_of_interest)
        continue
    instance_dict[token_of_interest] = len(segmentation_set)
with open("llama_instance_dict.pkl", "wb") as f:
    pkl.dump(instance_dict, f)
