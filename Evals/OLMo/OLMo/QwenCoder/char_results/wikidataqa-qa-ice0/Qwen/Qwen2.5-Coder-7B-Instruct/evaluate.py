import json
import re

# adjust this path to your file
INPUT_PATH = "predictions.jsonl"

# regex to grab the single letter after the assistant tag
total = 0
correct = 0
total_prompt_length = 0
with open(INPUT_PATH, "r") as f:
    for i, line in enumerate(f, 1):
        obj = json.loads(line)
        output = obj.get("output", "")
        gold_choice = obj.get("answer")
        total_prompt_length += len(obj.get("prompt"))
        if (type(gold_choice) == list):
            for item in gold_choice:
                if item.lower() in output.lower():
                    correct += 1
                    break
        else:
            if (gold_choice.lower() in output.lower()):
                correct += 1
        total += 1
print(correct / total)
print(total_prompt_length / total)