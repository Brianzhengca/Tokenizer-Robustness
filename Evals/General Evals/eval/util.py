import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi
import numpy as np
from pathlib import Path
from olmo.util import ensure_dir
import json
import pandas as pd
from methodtools import lru_cache
import random
from string import ascii_uppercase

class generateSegments:

    # Pre: string word representing the word to be segmented, set vocabulary representing 
    #       the vocabulary to check the segments against
    # Post: creates a new generateSegments Object, does not output.
    def __init__(self, vocabulary):
        self.vocab = vocabulary
    
    # Pre: accepts an integer start representing the index to start from
    # Post: returns the number of segments that potentially start from start index in self.word.
    #       Cached method. Uses methodtools.lru_cache instead of functools.cache to avoid 
    #       memory leaks
    @lru_cache()
    def countSegments(self, start):
        if start == self.wordlen:
            return 1
        total = 0
        for end in range(start + 1, self.wordlen + 1):
            if self.word[start:end] in self.vocab:
                total += self.countSegments(end)
        return total
    
    # Pre: accepts an integer start representing the starting index that we want to build 
    #       a segment from 
    # Post: builds the segmentation for self.word. At each recursive step, randomly choose 
    #       a child to visit, weighed by the size of the subtree rooted at the ith child
    def buildSegments(self, start):
        if start == self.wordlen:
            return []
        choices = []
        weights = []
        for end in range(start + 1, self.wordlen + 1):
            segment = self.word[start:end]
            if segment in self.vocab:
                count = self.countSegments(end)
                if count > 0:
                    choices.append(segment)
                    weights.append(count)
        if not choices:
            return []
        nextSegment = random.choices(choices, weights=weights, k=1)[0]
        return [nextSegment] + self.buildSegments(start + len(nextSegment))

    # Post: generate a random segmentation where each segment exists in self.vocab.
    #       Returns the segmentation
    def generate(self, word):
        self.word = word
        self.wordlen = len(word)
        total = self.countSegments(0)
        if total == 0:
            return [] # If there is no valid way to segment self.word 
        return self.buildSegments(0)



def prep_incontext_examples(test_df, num_incontext_examples):
    indices = np.arange(len(test_df))
    incontext_indices = {
        i: np.random.choice(indices[indices != i], size=num_incontext_examples, replace=False)
        for i in tqdm(indices, desc="Precomputing in-context examples")
    }
    return incontext_indices


def parse_number(output_str, output_type="int"):
    output_str = output_str.strip().replace(",", "")
    output_num = None
    try:
        if output_type == "int":
            output_num = int(output_str)
        elif output_type == "float":
            output_num = float(output_str)
    except ValueError:
        print(f"Failed to parse number: {output_str}")
        pass
    return output_num


def format_example(
    question, passage=None, choices=None, answer=None, qa_format="qnan", question_prefix="Question:"
):
    """Options for QA format:
    qa: Question: {question}\nAnswer: {answer}
    qnan: Question:\n{question}\nAnswer:\n{answer}
    qna: Question:\n{question}\nAnswer: {answer}
    q: Question: {question} (if answer=None, else equivalent to qa)
    """
    text = ""
    if passage:
        text += f"{passage.strip()}\n\n"

    text += question_prefix + "\n" if "qn" in qa_format else question_prefix + " "
    text += question.strip() + "\n"

    if choices:
        for label, choice in zip(ascii_uppercase, choices):
            text += f"{label}. {choice.strip()}\n"

    answer_prefix = "Answer:"
    if answer or qa_format != "q":
        text += answer_prefix + "\n" if "an" in qa_format else answer_prefix
    if answer:
        if isinstance(answer, str):
            answer = answer.strip()
        answer = str(answer)
        text += answer if "an" in qa_format else " " + answer

    return text


def parse_mc_pred(output, num_options=4, qa_format="qnan"):
    """
    Parses the predicted MC option (e.g., "A") from the model output.
    Returns None if the output is not a valid MC option.
    """
    parsed_answer = None
    valid = True
    if qa_format == "q":
        if output.startswith("Answer:"):  # output answer should start with "Answer: "
            output = output.replace("Answer: ", "")
        else:
            valid = False
    elif qa_format in ["qa", "qna"]:
        if output.startswith(" "):  # output answer should start with leading space
            output = output.lstrip()
        else:
            valid = False

    if output and valid and (output[0] in ascii_uppercase[:num_options]):
        parsed_answer = output[0]

    return parsed_answer


def get_checkpoints(model_name):
    refs = HfApi().list_repo_refs(model_name)
    checkpoints = []
    for branch in refs.branches:
        checkpoints.append(branch.name)
    return checkpoints


def batched_generate(prompts, model, tokenizer, batch_size=1, is_mcq=False, **generation_kwargs):
    def fiddle(inputTensor, attention_mask):
        inputList = inputTensor.tolist()
        fiddledPrompts = []
        if is_mcq:
            input_ids_prefix = tokenizer.encode(
                """<|im_start|>system\nYou are a helpful assistant. For the following multiple choice questions, return the answer only, without any additional reasoning or explanation. <|im_end|>\n<|im_start|>user\n""",
                add_special_tokens=False
            )
        else:
            input_ids_prefix = tokenizer.encode(
                """<|im_start|>system\nYou are a helpful assistant. For the following question, return the answer only, without any additional reasoning or explanation. <|im_end|>\n<|im_start|>user\n""",
                add_special_tokens=False
            )
        input_ids_suffix = tokenizer.encode(
            "<|im_end|>\n<|im_start|>assistant\n",
            add_special_tokens=False
        )
        for i, prompt in enumerate(inputList):
            fiddledPrompt = []
            realTokens = []
            tokenPrompt = tokenizer.convert_ids_to_tokens(prompt)
            for j, token in enumerate(tokenPrompt):
                if (attention_mask[i][j] == 1): # non-filler (real) tokens
                    realTokens.append(token)
            for token in realTokens:
                if token == tokenizer.bos_token:
                    continue
                fiddledPrompt += segmentor.generate(token)
                #fiddledPrompt += list(token)
            """fiddledPrompt = realTokens
            last = fiddledPrompt[-1]
            fiddledPrompt.pop()
            secondtolast = fiddledPrompt[-1]
            fiddledPrompt.pop()
            fiddledPrompt += list(secondtolast)
            fiddledPrompt.append(last)"""
            #fiddledPrompt = [tokenizer.bos_token] + fiddledPrompt + [tokenizer.eos_token]
            #print(fiddledPrompt)
            fiddledPrompts.append(input_ids_prefix + tokenizer.convert_tokens_to_ids(fiddledPrompt) + input_ids_suffix)
        #print(fiddledPrompts)
        #print(isinstance(fiddledPrompts[0], list) and isinstance(fiddledPrompts[0][0], int))
        return tokenizer.pad({"input_ids": fiddledPrompts}, padding="longest", padding_side="left", return_tensors="pt")
    generations = []
    segmentor = generateSegments(tokenizer.vocab.keys())
    pbar = tqdm(total=len(prompts), desc="Generating")
    batch_size = 5 # arbitrary
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        if is_mcq:
            batch_prompts = ["<s>[INST] You are a helpful assistant. For the following multiple choice questions, return the answer only, without any additional reasoning or explanation. " + prompt + "[/INST]" for prompt in batch_prompts]
        else:
            batch_prompts = ["<s>[INST] You are a helpful assistant. For the following question, return the answer only, without any additional reasoning or explanation. " + prompt + "[/INST]" for prompt in batch_prompts]
        #print(batch_prompts[0])
        # apply chat template here
        batch_inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            add_special_tokens=True,
            padding="longest",
        )
        res = fiddle(batch_inputs.input_ids, batch_inputs.attention_mask)
        batch_outputs = model.generate(
            **batch_inputs,
            num_return_sequences=1,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
            tokenizer=tokenizer,
            **generation_kwargs,
        )
        batch_generations = tokenizer.batch_decode(batch_outputs.sequences, skip_special_tokens=True)
        # remove the prompt from the generation
        #batch_generations = [gen[len(prompt) :] for prompt, gen in zip(batch_prompts, batch_generations)]
        #print(batch_generations)
        generations.extend(batch_generations)
        pbar.update(len(batch_prompts))
    return generations


def load_model_and_tokenizer(model_name_or_path, tokenizer_name_or_path=None, step=None, padding_side="left"):
    revision = None
    if os.path.exists(model_name_or_path):
        if step:
            model_name_or_path += f"/step{step}"
    else:
        if step:
            try:
                revision = [r for r in get_checkpoints(model_name_or_path) if r.split("-")[1] == f"step{step}"][0]
                print(f"Revision: {revision}")
            except IndexError:
                raise ValueError(f"Checkpoint {step} not found")

    tokenizer_name_or_path = tokenizer_name_or_path or model_name_or_path

    print(f"Loading model from {model_name_or_path}")

    # when model is too small, need to limit the number of visible devices
    # for some reason the device mapping doesn't work for small models on lots of GPUs
    if "1B" in model_name_or_path:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        revision=revision if "allenai" in model_name_or_path else None,
        force_download=True
    )
    model.eval()

    print(f"Loading tokenizer from {tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    tokenizer.backend_tokenizer.model.dropout = 0.0  # always use dropout p = 0.0 for inference
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = padding_side

    return model, tokenizer


def write_results(results, output_dir, metric="accuracy", print_metrics=False):
    metrics = {"num_examples": len(results), "accuracy": np.mean([r["correct"] for r in results])}

    if "valid" in results[0]:
        metrics["valid_answer"] = np.mean([r["valid"] for r in results])

    if "split" in results[0]:
        for split in sorted(set([r["split"] for r in results])):
            split_results = [r for r in results if r["split"] == split]
            metrics[f"{split}_accuracy"] = np.mean([r["correct"] for r in split_results])

    if print_metrics:
        for k, v in metrics.items():
            print(f"{k}: {v}")

    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    print(f"Saving results to {output_dir}")

    with open(output_dir / "metrics.json", "w") as fo:
        json.dump(metrics, fo, indent=4)
    with open(output_dir / "example_prompt.txt", "w") as fo:
        fo.write(results[0]["prompt"])
    pd.DataFrame(results).to_json(output_dir / "predictions.jsonl", orient="records", lines=True)
