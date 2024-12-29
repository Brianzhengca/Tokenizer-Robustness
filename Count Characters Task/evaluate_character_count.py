import re

groundtruth = open("character_level_groundtruth.txt", "r")
normalfile = open("normal_character_outputs.txt", "r")
perturbedfile = open("perturbed_character_outputs.txt", "r")

normallines = [line.rstrip() for line in normalfile.readlines()]
perturbedlines = [line.rstrip() for line in perturbedfile.readlines()]
groundtruthlines = [line.rstrip() for line in groundtruth.readlines()]

normalpredictions = []
perturbedpredictions = []
ground_truth = []

for i in range(len(normallines)):
    print(i)
    if (len(re.findall(r"[-+]?\d*\.\d+|\d+", normallines[i])) > 0 and len(re.findall(r"[-+]?\d*\.\d+|\d+", perturbedlines[i])) > 0):
        normal = re.findall(r"[-+]?\d*\.\d+|\d+", normallines[i])[-1]
        perturbed = re.findall(r"[-+]?\d*\.\d+|\d+", perturbedlines[i])[-1]
        normalpredictions.append(normal)
        perturbedpredictions.append(perturbed)
        ground_truth.append(groundtruthlines[i])

perturbedcorrect = 0
normalcorrect = 0
for index, perturbed in enumerate(perturbedpredictions):
    if perturbed == ground_truth[index]:
        perturbedcorrect += 1
    if normalpredictions[index] == ground_truth[index]:
        normalcorrect += 1
print("Normal:", normalcorrect / len(normalpredictions))
print("Perturbed:", perturbedcorrect / len(perturbedpredictions))

