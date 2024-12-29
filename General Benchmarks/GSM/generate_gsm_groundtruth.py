import json
import re

with open("full_gsm8k_test.jsonl", 'r') as file:
    data = [json.loads(line) for line in file]
groundtruthfile = open("gsm_groundtruth.txt", "w")
for line in data:
    groundtruthfile.write(re.findall(r"[-+]?\d*\.\d+|\d+", line["answer"])[-1] + "\n")

groundtruthfile.close()