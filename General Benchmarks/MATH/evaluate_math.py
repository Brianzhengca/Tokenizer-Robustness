from evaluation_utils import normalize_final_answer, remove_boxed, last_boxed_only_string
groundtruth = [line.rstrip() for line in open("math_groundtruth.txt", "r").readlines()]
normallines = [normalize_final_answer(remove_boxed(last_boxed_only_string((line.split("<|eot_id|>")[-2].split("<|end_header_id|>\\n\\n")[-1])))) for line in open("normal_math_outputs.txt", "r").readlines()]
perturbedlines = [normalize_final_answer(remove_boxed(last_boxed_only_string((line.split("<|eot_id|>")[-2].split("<|end_header_id|>\\n\\n")[-1])))) for line in open("perturbed_math_outputs.txt", "r").readlines()]
normalcorrect = 0
perturbedcorrect = 0
for index, answer in enumerate(groundtruth):
    if normallines[index] == answer:
        normalcorrect += 1
    if perturbedlines[index] == answer:
        perturbedcorrect += 1
print("Normal:", normalcorrect / len(groundtruth))
print("Perturbed:", perturbedcorrect / len(groundtruth))
