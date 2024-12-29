import re
groundtruth = [line.rstrip() for line in open("codeline_groundtruth.txt", "r").readlines()]
normallines = [line.split("<|eot_id|>")[-2].split("<|end_header_id|>\\n\\n")[-1] for line in open("normal_codeline_outputs.txt", "r").readlines()]
perturbedlines = [line.split("<|eot_id|>")[-2].split("<|end_header_id|>\\n\\n")[-1] for line in open("perturbed_codeline_outputs.txt", "r").readlines()]
print(perturbedlines)
perturbedcorrect = 0
normalcorrect = 0
for i in range(len(groundtruth)):
    if re.search(r"(?:\s|^)[A-D](?=\s|[.,!?]|$)", perturbedlines[i]) != None:
        if groundtruth[i] == re.search(r'(?:\s|^)[A-D](?=\s|[.,!?]|$)', perturbedlines[i]).group().strip():
            perturbedcorrect += 1
    if re.search(r"(?:\s|^)[A-D](?=\s|[.,!?]|$)", normallines[i]) != None:
        if groundtruth[i] == re.search(r'(?:\s|^)[A-D](?=\s|[.,!?]|$)', normallines[i]).group().strip():
            normalcorrect += 1
print("Normal:", normalcorrect / len(groundtruth))
print("Perturbed:", perturbedcorrect / len(groundtruth))