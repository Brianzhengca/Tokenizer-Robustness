import json

promptfile = open("gsm_prompts.txt", "w")
inputfile = open("full_gsm8k_test.jsonl", "r")

lines = [json.loads(line) for line in inputfile.readlines()]

for line in lines:
    promptfile.write(line["question"] + "\n")

promptfile.close()
inputfile.close()