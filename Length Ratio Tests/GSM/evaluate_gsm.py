import json
import re
import pickle as pkl
import pandas as pd

datapoints = [] # [length_ratio, accuracy_mask, id]

for i in range(2, 7):
    with open("full_gsm8k_test.jsonl", 'r') as file:
        data = [json.loads(line) for line in file]
    ground_truth = []
    perturbedpredictions = []
    length_ratio = str(i)
    with open("math_length" + length_ratio + ".0_ratios.pkl", "rb") as f:
        ratios = pkl.load(f)
    #print(ratios)
    perturbedfile = open("perturbed_gsm" + length_ratio + ".0_outputs.txt", "r")
    perturbedlines = [line.rstrip() for line in perturbedfile.readlines()]
    for k in range(len(perturbedlines)):
        if (len(re.findall(r"[-+]?\d*\.\d+|\d+", perturbedlines[k])) > 0):
            perturbed = re.findall(r"[-+]?\d*\.\d+|\d+", perturbedlines[k])[-1]
            perturbedpredictions.append(perturbed)
        else:
            perturbedpredictions.append(float('inf')) #perturbed prediction produced no numbers, is wrong
    for row in data:
        ground_truth.append(re.findall(r"[-+]?\d*\.\d+|\d+", row["answer"])[-1])
    perturbedpredictions = [float(item) for item in perturbedpredictions]
    ground_truth = [float(item) for item in ground_truth]
    for j in range(len(ground_truth)):
        if (perturbedpredictions[j] == ground_truth[j]):
            #print(perturbedpredictions[j], ground_truth[j])
            datapoints.append([ratios[j], 1, j])
        else:
            datapoints.append([ratios[j], 0, j])
df = pd.DataFrame(datapoints, columns=['length_ratio', 'accuracy_mask', 'id'])
intervals = {
    'df_1_2': (1, 2),
    'df_2_3': (2, 3),
    'df_3_4': (3, 4),
    'df_4_5': (4, 5)
}
masks = {
    name: (df['length_ratio'] >= lower) & (df['length_ratio'] < upper)
    for name, (lower, upper) in intervals.items()
}
masks['df_4_5'] = (df['length_ratio'] >= 4) & (df['length_ratio'] <= 5)
df_1_2 = df[masks['df_1_2']].reset_index(drop=True)
df_2_3 = df[masks['df_2_3']].reset_index(drop=True)
df_3_4 = df[masks['df_3_4']].reset_index(drop=True)
df_4_5 = df[masks['df_4_5']].reset_index(drop=True)
ids_1_2 = set(df_1_2['id'].unique())
ids_2_3 = set(df_2_3['id'].unique())
ids_3_4 = set(df_3_4['id'].unique())
ids_4_5 = set(df_4_5['id'].unique())
common_ids = ids_1_2 & ids_2_3 & ids_3_4 & ids_4_5
print(len(common_ids))
df_1_2_filtered = df_1_2[df_1_2['id'].isin(common_ids)].reset_index(drop=True)
df_2_3_filtered = df_2_3[df_2_3['id'].isin(common_ids)].reset_index(drop=True)
df_3_4_filtered = df_3_4[df_3_4['id'].isin(common_ids)].reset_index(drop=True)
df_4_5_filtered = df_4_5[df_4_5['id'].isin(common_ids)].reset_index(drop=True)
df_1_2_filtered = df_1_2_filtered.drop_duplicates(subset=["id"], keep="first")
df_2_3_filtered = df_2_3_filtered.drop_duplicates(subset=["id"], keep="first")
df_3_4_filtered = df_3_4_filtered.drop_duplicates(subset=["id"], keep="first")
df_4_5_filtered = df_4_5_filtered.drop_duplicates(subset=["id"], keep="first")
print(df_1_2_filtered["accuracy_mask"].mean())
print(df_2_3_filtered["accuracy_mask"].mean())
print(df_3_4_filtered["accuracy_mask"].mean())
print(df_4_5_filtered["accuracy_mask"].mean())
print(len(df_1_2_filtered))
print(len(df_2_3_filtered))
print(len(df_3_4_filtered))
print(len(df_4_5_filtered))