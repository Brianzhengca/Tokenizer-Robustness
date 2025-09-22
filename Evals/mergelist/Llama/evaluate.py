import re
import json

from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.stats import kendalltau

import seaborn as sns
import numpy as np

from matplotlib import font_manager
import matplotlib.pyplot as plt

font_dir = ["/mmfs1/gscratch/ark/zhengbr/tokenizer-robustness/fonts"]
for font in font_manager.findSystemFonts(font_dir):
    font_manager.fontManager.addfont(font)

plt.rcParams.update(
    {
        "font.family": "Manrope",
        "axes.linewidth": 1.5,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#333333",
        "axes.titlecolor": "#333333",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "legend.labelcolor": "#333333",
        "legend.fontsize": 20,
        "text.color": "#333333",
        "font.size": 26,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
    }
)

def evaluate_mcq(item):
    mcq_re = re.compile(r"assistant\s*\n\s*([A-E])")
    total = 0
    correct = 0
    for i, line in enumerate(item):
        m = mcq_re.search(line["output"])
        assistant_choice = m.group(1) if m else None
        groundtruth = line["answer"]
        if (assistant_choice == groundtruth):
            correct += 1
        total += 1
    return correct / total

def evaluate_boolq(item):
    mcq_re = re.compile(r"assistant\s*\n\s*(Yes|No)\b")
    total = 0
    correct = 0
    for i, line in enumerate(item):
        m = mcq_re.search(line["output"])
        assistant_choice = m.group(1) if m else None
        groundtruth = line["answer"]
        if (assistant_choice == groundtruth):
            correct += 1
        total += 1
    return correct / total

def evaluate_default_sa(item):
    total = 0
    correct = 0
    for i, line in enumerate(item):
        if line["correct"]:
            correct += 1
        total += 1
    return correct / total

def evaluate_custom_sa(item):
    total = 0
    correct = 0
    for i, line in enumerate(item):
        output = line["output"]
        answer = line["answer"]
        if (answer.lower() in output.lower()):
            correct += 1
        total += 1
    return correct / total

def evaluate_wikidata(item):
    correct = 0
    total = 0
    for i, line in enumerate(item):
        output = line["output"]
        groundtruth = line["answer"]
        if (type(groundtruth) == list):
            for option in groundtruth:
                if (option.lower() in output.lower()):
                    correct += 1
                    break
        else:
            if groundtruth.lower() in output.lower():
                correct += 1
        total += 1
    return correct / total

path_map = {
    "ARC Challenge" : "arc-challenge-qa-ice0",
    "ARC Easy" : "arc-easy-qa-ice0",
    "BoolQ" : "boolq-qa-ice0",
    "CommonSenseQA" : "commonsenseqa-qa-ice0",
    "COPA" : "copa-qa-ice0",
    "CUTE" : "cute-qa-ice0",
    "OpenBookQA" : "openbookqa-qa-ice0",
    "HellaSwag" : "hellaswag-qa-ice0",
    "MMLU" : "mmlu-qa-ice0",
    "PIQA" : "piqa-qa-ice0",
    "Winograd" : "winograd-qa-ice0",
    "WinoGrande" : "winogrande-qa-ice0",
    "TriviaQA" : "triviaqa-qa-ice0",
    "WikidataQA" : "wikidataqa-qa-ice0",
    "CUTE" : "cute-qa-ice0",
    "JeopardyQA" : "jeopardy-qa-ice0",
    "DROP" : "drop-qa-ice0",
    "TOFU" : "tofu-qa-ice0",
    "Winograd" : "winograd-qa-ice0",
    "WinoGrande" : "winogrande-qa-ice0",
}

eval_funcs = {
    "ARC Challenge" : evaluate_mcq,
    "ARC Easy" : evaluate_mcq,
    "BoolQ" : evaluate_boolq,
    "CommonSenseQA" : evaluate_mcq,
    "COPA" : evaluate_mcq,
    "CUTE" : evaluate_custom_sa,
    "HellaSwag" : evaluate_mcq,
    "MMLU" : evaluate_mcq,
    "PIQA" : evaluate_mcq,
    "Winograd" : evaluate_mcq,
    "WinoGrande" : evaluate_mcq,
    "TriviaQA" : evaluate_default_sa,
    "CUTE" : evaluate_custom_sa,
    "DROP" : evaluate_default_sa,
}

def main(target_eval, eval_func):
    paths = ["normal_results", "1.5-2.0", "2.0-2.5", "2.5-3.0", "3.0-3.5", "3.5-4.0"]
    bins = []

    for path in paths:
        target_path = "./" + path + "/" + path_map[target_eval] + "/meta-llama/Llama-3.1-8B-Instruct/predictions.jsonl"
        predictions = []
        with open(target_path, "r") as f:
            for line in f.readlines():  
                predictions.append(json.loads(line))
        bins.append(predictions)

    valid_indices = set()
    for i in range(len(bins[0])):
        correct = True
        for bin in bins:
            if ("IMPOSSIBLE INPUT" in bin[i]["output"]):
                correct = False
                break
        if correct:
            valid_indices.add(i)
    for index, bin in enumerate(bins):
        new_bin = []
        for i in range(len(bin)):
            if (i in valid_indices):
                new_bin.append(bin[i])
        bins[index] = new_bin
    res = [eval_func(bin) for bin in bins]
    print(target_eval, res, len(bins[0]))
    return {target_eval:res}
labels = ["Canonical", "1.5-2.0", "2.0-2.5", "2.5-3.0", "3.0-3.5", "3.5-4.0"]
data = [{"Alpaca Eval":[0.500, 0.498, 0.470, 0.433, 0.418, 0.420]}, {"GSM8K":[0.820, 0.830, 0.750, 0.707, 0.737, 0.667]}]
for key, value in tqdm(eval_funcs.items()):
    data.append(main(key, value))

series = {name: values for d in data for name, values in d.items()}

normalized = {
    name: [v / values[0] * 100 for v in values]
    for name, values in series.items()
}

plt.figure(figsize=(20, 8))
for name, values in normalized.items():
    sns.lineplot(x=labels, y=values, marker='o', label=name)

all_values = np.array(list(normalized.values()))   
avg_values = all_values.mean(axis=0)  
median_values = np.median(all_values, axis=0)  

corr, p = kendalltau(np.array(list(range(len(avg_values)))), avg_values)
print(corr, p)

sns.lineplot(
    x=labels,
    y=avg_values,
    marker='o',
    linewidth=5,          
    color='black',        
    label='Mean Performance'
)

plt.xlabel('Length Ratio')
plt.ylabel('Relative Change')
#plt.title('Fine vs Coarse Segmentation')
plt.legend(ncol=2, title='Benchmark', fontsize='small', title_fontsize='medium', loc='center left', bbox_to_anchor=(1.02, 0.5))
plt.tight_layout()
plt.savefig("result.jpg", dpi=500)
