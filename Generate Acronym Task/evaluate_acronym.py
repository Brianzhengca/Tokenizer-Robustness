groundtruth = [line.rstrip() for line in open("acronym_groundtruth.txt", "r").readlines()]
normallines = [line.split("<|eot_id|>")[-2].split("<|end_header_id|>\\n\\n")[-1] for line in open("normal_acronym_outputs.txt", "r").readlines()]
perturbedlines = [line.split("<|eot_id|>")[-2].split("<|end_header_id|>\\n\\n")[-1] for line in open("perturbed_acronym_outputs.txt", "r").readlines()]
perturbedcorrect = 0
normalcorrect = 0
for i in range(len(normallines)):
    perturbedline = perturbedlines[i].split()
    perturbedacronym = ''.join([word[0] for word in perturbedline]).lower()
    normalline = normallines[i].split()
    normalacronym = ''.join(word[0] for word in normalline).lower()
    print(perturbedline, normalline)
    if groundtruth[i] == perturbedacronym:
        perturbedcorrect += 1
    if groundtruth[i] == normalacronym:
        normalcorrect += 1
print("Perturbed:", perturbedcorrect / len(groundtruth))
print("Normal:", normalcorrect / len(groundtruth))