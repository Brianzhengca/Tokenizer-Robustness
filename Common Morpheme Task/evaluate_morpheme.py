import re
groundtruth = [line.rstrip() for line in open("morpheme_groundtruth.txt", "r").readlines()]
normallines = [line.split("<|eot_id|>")[-2].split("<|end_header_id|>\\n\\n")[-1] for line in open("normal_morpheme_outputs.txt", "r").readlines()]
perturbedlines = [line.split("<|eot_id|>")[-2].split("<|end_header_id|>\\n\\n")[-1] for line in open("perturbed_morpheme_outputs.txt", "r").readlines()]
perturbedcorrect = 0
normalcorrect = 0
for i in range(len(groundtruth)):
    print(groundtruth[i], re.search(r'[A-Z]', perturbedlines[i]).group(), re.search(r'[A-Z]', normallines[i]).group())
    if groundtruth[i] == re.search(r'[A-Z]', perturbedlines[i]).group():
        perturbedcorrect += 1
    if (groundtruth[i] == re.search(r'[A-Z]', normallines[i]).group()):
        normalcorrect += 1
print("Normal:", normalcorrect / len(groundtruth))
print("Perturbed:", perturbedcorrect / len(groundtruth))