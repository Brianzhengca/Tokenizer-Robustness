import json
import re

# adjust this path to your file
INPUT_PATH = "predictions.jsonl"

# regex to grab the single letter after the assistant tag
assistant_re = re.compile(r"(\d+)(?!.*\d)")
total = 0
correct = 0
with open(INPUT_PATH, "r") as f:
    for i, line in enumerate(f, 1):
        obj = json.loads(line)
        output = obj.get("output", "")
        # split/regex combo to find the assistant’s choice
        m = assistant_re.search(output)
        assistant_choice = m.group(1) if m else None
        # ground-truth
        gold_choice = obj.get("answer")
        print(assistant_choice, gold_choice)
        if (assistant_choice == gold_choice):
            correct += 1
        total += 1
print(correct / total)