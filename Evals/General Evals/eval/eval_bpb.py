import torch
import torch.nn.functional as F
import json
import os
import random
import pandas as pd
import click
from tqdm import tqdm
from pathlib import Path
import regex as re
from olmo.util import ensure_dir
from eval.util import load_model_and_tokenizer

random.seed(42)
pretok_pattern = r"(?=(\\d{3})+(?!\\d))| ?\p{L}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"


def get_bits_per_byte(model, tokenizer, eval_data, batch_size, max_context_length):
    total_accuracy = 0
    total_first_word_correct = 0
    total_label_pred_has_first_word = 0
    total_first_byte_correct = 0
    total_loss = 0
    total_tokens = 0
    total_bytes = 0

    examples = []
    for i in tqdm(range(0, len(eval_data), batch_size)):
        batch_texts = eval_data[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_context_length,
        )
        inputs.attention_mask = inputs.attention_mask.to(model.device)
        inputs.input_ids = inputs.input_ids.to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)
            logits = outputs.logits.to(model.device)
            labels = inputs.input_ids.clone()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask = inputs.attention_mask[..., :-1].contiguous()

            # first word correct: whether the first whitespace-delimited word of the prediction matches the ground truth
            # first byte correct: whether the first char of the prediction matches the ground truth
            predictions = shift_logits.argmax(dim=-1)
            # decode predictions and shift_labels to see if the strings match
            for pred_seq, label_seq in zip(predictions.tolist(), shift_labels.tolist()):
                for pred_token_id, label_token_id in zip(pred_seq, label_seq):  # iterate over tokens
                    if label_token_id == tokenizer.pad_token_id:
                        continue

                    ex = {"first_byte_correct": False, "first_word_correct": False}
                    pred_token = tokenizer.decode([pred_token_id])
                    label_token = tokenizer.decode([label_token_id])

                    # compare first char
                    if pred_token[0] == label_token[0]:
                        total_first_byte_correct += 1
                        ex["first_byte_correct"] = True

                    # use regex to split into subwords
                    pred_subwords = [match.group() for match in re.finditer(pretok_pattern, pred_token)]
                    label_subwords = [match.group() for match in re.finditer(pretok_pattern, label_token)]
                    ex["pred"] = pred_token
                    ex["label"] = label_token

                    pred_first_subword = pred_subwords[0] if pred_subwords else None
                    label_first_subword = label_subwords[0] if label_subwords else None
                    ex["pred_first_subword"] = pred_first_subword
                    ex["label_first_subword"] = label_first_subword

                    # if both have first subword, check if they match
                    if pred_first_subword and label_first_subword:
                        total_label_pred_has_first_word += 1
                        if pred_first_subword == label_first_subword:
                            total_first_word_correct += 1
                            ex["first_word_correct"] = True
                    examples.append(ex.copy())

        correct = (predictions == shift_labels).float() * shift_attention_mask
        total_accuracy += correct.sum()

        # get neg log likelihood
        loss = F.cross_entropy(shift_logits.transpose(1, 2), shift_labels, reduction="none") * shift_attention_mask
        total_loss += loss.sum()
        total_tokens += shift_attention_mask.sum()

        total_bytes += sum(
            len(text) for text in tokenizer.batch_decode(inputs.input_ids, skip_special_tokens=True)
        )

    bits_per_byte = total_loss / total_bytes / torch.log(torch.tensor(2.0))
    accuracy = total_accuracy / total_tokens
    first_word_correct = total_first_word_correct / total_label_pred_has_first_word
    first_byte_correct = total_first_byte_correct / total_tokens
    metrics = {
        "bits_per_byte": bits_per_byte.item(),
        "accuracy": accuracy.item(),
        "first_word_correct": first_word_correct,
        "first_byte_correct": first_byte_correct.item(),
        "total_tokens": total_tokens.item(),
        "total_bytes": total_bytes,
        "num_examples": len(eval_data),
        "max_context_length": max_context_length,
    }

    return metrics, examples


@click.command()
@click.option("--model_name_or_path", type=str, default="OLMo2-7B-npt200k")
@click.option("--step", type=int, default=None)
@click.option("--output_dir", type=str, default="results/bpb/olmo-20k")
@click.option(
    "--eval_dir",
    type=str,
    help="Path to folder with jsonl files",
    default="olmo_data/olmo2_shuffle/olmo2_shuffle",
)
@click.option("--max_num_examples", type=int, default=None)
@click.option("--eval_batch_size", type=int, default=8)
@click.option("--add_bos_token", is_flag=True, default=False)
@click.option("--max_context_length", type=int, default=None)
def main(
    model_name_or_path: str,
    step: int,
    output_dir: str,
    eval_dir: str,
    max_num_examples: int,
    eval_batch_size: int,
    add_bos_token: bool,
    max_context_length: int,
):
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, step=step, add_bos_token=add_bos_token)
    max_context_length = max_context_length or model.config.max_position_embeddings
    print(f"Using max context length of {max_context_length}")

    # read .jsonl files from eval_dir until we have max_num_examples
    eval_data = []
    for eval_file in os.listdir(eval_dir):
        nrows = max_num_examples - len(eval_data)
        df = pd.read_json(os.path.join(eval_dir, eval_file), lines=True, compression="zstd", nrows=nrows)
        eval_data.extend(df.text.tolist())
        if len(eval_data) >= max_num_examples:
            break

    eval_data = random.sample(eval_data, max_num_examples)
    print(f"Loaded {len(eval_data)} examples.")

    metrics, examples = get_bits_per_byte(
        model, tokenizer, eval_data=eval_data, batch_size=eval_batch_size, max_context_length=max_context_length
    )

    # print metrics
    for k, v in metrics.items():
        print(f"{k}: {v}")

    if add_bos_token:
        output_dir += "-bos"
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    print(f"Saving results to {output_dir}")

    with open(output_dir / "metrics.json", "w") as fo:
        json.dump(metrics, fo, indent=4)
    pd.DataFrame(examples).to_json(output_dir / "predictions.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    main()
