from batchedGeneration import TransformersInterface
from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle as pkl

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

f = open("mmlu_prompts.txt", "r", errors="ignore")
lines = []
for line in f.readlines():
    lines.append(line.rstrip())
interface = TransformersInterface(model, vocabulary, tokenizer)
for i in range(1, 6):
    length_ratio = 1 + i/1
    print("LENGTH RATIO:", length_ratio)
    normal_output_file = open("normal_mmlu" + str(length_ratio) + "_outputs.txt", "w")
    perturbed_output_file = open("perturbed_mmlu" + str(length_ratio) + "_outputs.txt", "w")
    normal_responses, perturbed_responses, length_ratios = interface.generate_response(lines, max_new_tokens=2048, batch_size=2, target_length = length_ratio)
    for i in range(len(normal_responses)):
        normal_output_file.write(str([normal_responses[i]]) + "\n")
        perturbed_output_file.write(str([perturbed_responses[i]]) + "\n")
        print("Processed:", i)
    normal_output_file.close()
    perturbed_output_file.close()
    f.close()
    with open("mmlu_length" + str(length_ratio) + "_ratios.pkl", "wb") as f:
        pkl.dump(length_ratios, f)
