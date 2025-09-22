import json
import re

INPUT_PATH = "predictions.jsonl"

correct = 0
total = 0

assistant_re = re.compile(r"Answer:\s*(Yes|No)\b")
prompt_length_sum = 0
with open(INPUT_PATH, "r") as f:
    for i, line in enumerate(f, 1):
        obj = json.loads(line)
        output = obj.get("output", "")
        prompt = obj.get("prompt")
        prompt_length_sum += len(prompt)
        m = assistant_re.search(output)
        assistant_choice = m.group(1) if m else None
        if not assistant_choice: continue
        gold_choice = obj.get("answer")
        if (assistant_choice == gold_choice):
            correct += 1
        total += 1

print(correct/total)
print(prompt_length_sum / total)