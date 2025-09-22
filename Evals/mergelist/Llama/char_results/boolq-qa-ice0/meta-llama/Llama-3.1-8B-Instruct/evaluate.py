import json
import re

INPUT_PATH = "predictions.jsonl"

correct = 0
total = 0

assistant_re = re.compile(r"assistant\s*\n\s*(Yes|No)\b")

with open(INPUT_PATH, "r") as f:
    for i, line in enumerate(f, 1):
        obj = json.loads(line)
        output = obj.get("output", "")
        m = assistant_re.search(output)
        assistant_choice = m.group(1) if m else None
        gold_choice = obj.get("answer")
        if (assistant_choice == gold_choice):
            correct += 1
        total += 1

print(correct/total)
