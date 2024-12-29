from datasets import load_dataset
import json

normallines = [line.rstrip() for line in open("normal_alpaca_outputs.txt", "r")]
perturbedlines = [line.rstrip() for line in open("perturbed_alpaca_outputs.txt", "r")]
eval_set = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", trust_remote_code=True)["eval"]
normal_alpaca_inputs = []
perturbed_alpaca_inputs = []
for i in range(len(normallines)):
    perturbed = eval(perturbedlines[i])[0].split("<|end_header_id|>")[-1].split("<|eot_id|>")[0]
    normal = eval(normallines[i])[0].split("<|end_header_id|>")[-1].split("<|eot_id|>")[0]
    prompt = eval_set[i]['instruction']
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
        "generator":"Normal",
        "dataset":"AlpacaEval_Default",
        "datasplit":"eval"
    }
    normal_alpaca_inputs.append(normalelement)
    perturbed_alpaca_inputs.append(perturbedelement)
with open("alpaca_reference.json", "w") as f:
    json.dump(normal_alpaca_inputs, f)
with open("alpaca_target.json", "w") as f:
    json.dump(perturbed_alpaca_inputs, f)
