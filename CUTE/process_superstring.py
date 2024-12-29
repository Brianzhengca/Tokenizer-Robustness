import os
import pickle as pkl
import base64

def encode_filename(token):
    # Encode as bytes first:
    token_bytes = token.encode("utf-8")
    # Base64 encode:
    encoded = base64.urlsafe_b64encode(token_bytes).decode("utf-8")
    # This ensures only [A-Za-z0-9_-] characters are used.
    return encoded

with open("instance_dict.pkl", "rb") as f:
    instance_dict = pkl.load(f)
superstring_dir = "olmo_superstring_encodings"
paths = os.listdir(superstring_dir)
promptfile = open("spelling_prompts.txt", "w")
groundtruthfile = open("spelling_groundtruth.txt", "w")

instance_dict = {k: v for k, v in sorted(instance_dict.items(), key=lambda item: item[1])}
for index, (key, value) in enumerate(instance_dict.items()):
    print(index)
    occurrence = sum(1 for _ in open(f"olmo_superstring_encodings/{key}.jsonl"))
    if ((value == 1 and 5 <= len(key) <= 15) or (value > 5 and 5 <= len(key) <= 15)) and 100 < occurrence < 500:
        print(value, occurrence)
        promptfile.write("Spell the following string: '" + key + "'\n")
        groundtruthfile.write(str(key) + " " + str(value) + "\n")
promptfile.close()
groundtruthfile.close()