import re
normallines = [line.rstrip() for line in open("normal_gsm_outputs.txt", "r").readlines()]
perturbedlines = [line.rstrip() for line in open("perturbed_gsm_outputs.txt", "r").readlines()]
groundtruths = [float(line.rstrip()) for line in open("gsm_groundtruth.txt", "r").readlines()]
perturbedcorrect = 0
normalcorrect = 0
for i in range(len(groundtruths)):
    if (len(re.findall(r"[-+]?\d*\.\d+|\d+", perturbedlines[i])) > 0):
        perturbed = float(re.findall(r"[-+]?\d*\.\d+|\d+", perturbedlines[i])[-1])
    else:
        perturbed = float('inf')
    if (len(re.findall(r"[-+]?\d*\.\d+|\d+", normallines[i])) > 0):
        normal = float(re.findall(r"[-+]?\d*\.\d+|\d+", normallines[i])[-1])
    else:
        normal = float('inf')
    if perturbed == groundtruths[i]:
        perturbedcorrect += 1
    if normal == groundtruths[i]:
        normalcorrect += 1
print("Normal:", normalcorrect / len(groundtruths))
print("Perturbed:", perturbedcorrect / len(groundtruths))