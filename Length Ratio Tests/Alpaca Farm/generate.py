from batchedGeneration import TransformersInterface
from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle as pkl
import datasets

eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", trust_remote_code=True)["eval"]
# Initialize the tokenizer and model
model_name = 'meta-llama/Llama-3.1-8B-Instruct'  # Ensure this is the correct model name
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token="your hf token")
# Ensure the tokenizer has a pad token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

pad_token_id = tokenizer.pad_token_id

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, token="your hf token")

# Extract vocabulary from tokenizer
vocabulary = set(tokenizer.get_vocab().keys())

lines = []
for line in eval_set:
    lines.append(line["instruction"])
interface = TransformersInterface(model, vocabulary, tokenizer)
for i in range(4, 5):
    length_ratio = 1 + i
    print("LENGTH RATIO:", length_ratio)
    normal_output_file = open("normal_alpaca" + str(length_ratio) + "_outputs.txt", "w")
    perturbed_output_file = open("perturbed_alpaca" + str(length_ratio) + "_outputs.txt", "w")
    normal_responses, perturbed_responses, length_ratios = interface.generate_response(lines, max_new_tokens=512, batch_size=5, target_length = length_ratio)
    for i in range(len(normal_responses)):
        normal_output_file.write(str([normal_responses[i]]) + "\n")
        perturbed_output_file.write(str([perturbed_responses[i]]) + "\n")
        print("Processed:", i)
    normal_output_file.close()
    perturbed_output_file.close()
    with open("alpaca_length" + str(length_ratio) + "_ratios.pkl", "wb") as f:
        pkl.dump(length_ratios, f)
