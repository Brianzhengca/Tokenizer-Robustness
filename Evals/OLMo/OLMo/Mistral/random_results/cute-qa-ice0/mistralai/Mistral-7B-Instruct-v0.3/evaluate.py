import json
import re

# adjust this path to your file
INPUT_PATH = "predictions.jsonl"

total = 0
correct = 0
total_length_sum = 0
with open(INPUT_PATH, "r") as f:
    for i, line in enumerate(f, 1):
        obj = json.loads(line)
        output = obj.get("output", "")
        total_length_sum += len(obj.get("prompt"))
        gold_choice = obj.get("answer")
        if (gold_choice.lower() in output.lower()):
            correct += 1
        total += 1
print(correct / total)
print(total_length_sum / total)