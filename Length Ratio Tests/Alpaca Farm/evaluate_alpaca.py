import pandas as pd
import pickle as pkl
import json

from datasets import load_dataset

eval_set = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]['instruction']

df = [] # [[id, generation, length_ratio]]

for i in range(2, 6):
    index = str(i)
    with open("alpaca_length" + index + "_ratios.pkl", "rb") as f:
        ratios = pkl.load(f)
    perturbed_file = open("perturbed_alpaca" + index + "_outputs.txt", "r")
    normal_file = open("normal_alpaca" + index + "_outputs.txt", "r")
    perturbed_lines = [line.rstrip() for line in perturbed_file.readlines()]
    normal_lines = [line.rstrip() for line in normal_file.readlines()]
    for j in range(len(ratios)):
        df.append([j, perturbed_lines[j], normal_lines[j], ratios[j]])
    perturbed_file.close()
    normal_file.close()
df = pd.DataFrame(df, columns=['id', 'perturbed_generation', 'normal_generation', 'length_ratio'])
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
df_1_2_filtered = df_1_2[df_1_2['id'].isin(common_ids)].reset_index(drop=True)
df_2_3_filtered = df_2_3[df_2_3['id'].isin(common_ids)].reset_index(drop=True)
df_3_4_filtered = df_3_4[df_3_4['id'].isin(common_ids)].reset_index(drop=True)
df_4_5_filtered = df_4_5[df_4_5['id'].isin(common_ids)].reset_index(drop=True)

def generate_alpacafarm_outputs(df, index):
    normal_alpaca_inputs = []
    perturbed_alpaca_inputs = []
    for i, row in df.iterrows():
        perturbed = eval(row["perturbed_generation"])[0].split("<|end_header_id|>")[-1].split("<|eot_id|>")[0]
        normal = eval(row["normal_generation"])[0].split("<|end_header_id|>")[-1].split("<|eot_id|>")[0]
        prompt = eval_set[row["id"]]
        print(prompt)
        normalelement = {
            "instruction":prompt,
            "output":normal,
            "generator":"Normal",
            "dataset":"AlpacaEval_Default",
            "datasplit":"eval"
        }
        perturbedelement = {
            "instruction":prompt,
            "output":perturbed,
            "generator":"Perturbed",
            "dataset":"AlpacaEval_Default",
            "datasplit":"eval"
        }
        normal_alpaca_inputs.append(normalelement)
        perturbed_alpaca_inputs.append(perturbedelement)
    print(len(perturbed_alpaca_inputs), len(normal_alpaca_inputs))
    with open(str(index) + "_alpaca_reference.json", "w") as f:
        json.dump(normal_alpaca_inputs, f)
    with open(str(index) + "_alpaca_target.json", "w") as f:
        json.dump(perturbed_alpaca_inputs, f)
generate_alpacafarm_outputs(df_1_2_filtered, "1-2")
generate_alpacafarm_outputs(df_2_3_filtered, "2-3")
generate_alpacafarm_outputs(df_3_4_filtered, "3-4")
generate_alpacafarm_outputs(df_4_5_filtered, "4-5")
print(df_1_2_filtered)
print(df_2_3_filtered)
print(df_3_4_filtered)
print(df_4_5_filtered)