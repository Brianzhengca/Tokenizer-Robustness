from datasets import load_dataset

ds = load_dataset("cais/mmlu", "all")["test"]
groundtruthfile = open("mmlu_groundtruth.txt", "w")
answerchoices = "ABCD"
for row in ds:
    groundtruthfile.write(answerchoices[row["answer"]] + "\n")
groundtruthfile.close()