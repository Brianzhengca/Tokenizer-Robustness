import json
from evaluation_utils import normalize_final_answer, remove_boxed, last_boxed_only_string

promptfile = open("math_prompts.txt", "w")
groundtruthfile = open("math_groundtruth.txt", "w")
prompts = [json.loads(line) for line in open("test.jsonl", "r").readlines()]

def prune_solution(full_solution):
    return normalize_final_answer(remove_boxed(last_boxed_only_string((full_solution))))

for prompt in prompts:
    promptfile.write(prompt["problem"].replace("\n", " ") + "\n")
    groundtruthfile.write(prune_solution(prompt["solution"]) + "\n")

promptfile.close()
groundtruthfile.close()
