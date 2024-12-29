from datasets import load_dataset
from evaluation_utils import normalize_final_answer, remove_boxed, last_boxed_only_string

import re
import random

ds = load_dataset("lighteval/MATH", "all")
promptfile = open('sampled_math_prompts.txt', 'w')
groundtruthfile = open("sampled_math_groundtruth.txt", "w")

def prune_solution(full_solution):
    return normalize_final_answer(remove_boxed(last_boxed_only_string((full_solution))))

fullprompts = []
fullgroundtruths = []

for row in ds["test"]:
    fullprompts.append(row["problem"].replace("\n", " ") + "\n")
    fullgroundtruths.append(prune_solution(row["solution"].replace("\\", "\\\\")) + "\n")

sampled_prompts, sampled_groundtruth = zip(*random.sample(list(zip(fullprompts, fullgroundtruths)), 800))
for index, prompt in enumerate(sampled_prompts):
    promptfile.write(prompt)
    groundtruthfile.write(sampled_groundtruth[index])

promptfile.close()
groundtruthfile.close()