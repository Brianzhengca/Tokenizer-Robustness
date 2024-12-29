from datasets import load_dataset

ds = load_dataset("cais/mmlu", "all")["test"]
promptfile = open("mmlu_prompts.txt", "w")
groundtruthfile = open("mmlu_groundtruth.txt", "w")

for row in ds:
    choices = row["choices"]
    prompt = row["question"].replace("\n", " ") + " A)" + choices[0] + " B)" + choices[1] + " C)" + choices[2] + " D)" + choices[3] + "\n"
    promptfile.write(prompt)
promptfile.close()
groundtruthfile.close()
