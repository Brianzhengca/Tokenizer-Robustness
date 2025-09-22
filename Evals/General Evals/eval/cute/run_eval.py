"""
Download data from https://huggingface.co/datasets/leukas/cute
"""

import re
import os

import click
import pandas as pd

from eval.util import (
    batched_generate,
    format_example,
    load_model_and_tokenizer,
    prep_incontext_examples,
    write_results,
)
from olmo.util import seed_all

seed_all(42)


def evaluate_cute(model, tokenizer, test_df, batch_size, num_incontext_examples, qa_format, split):
    test_df = test_df.reset_index(drop=True)
    incontext_indices = prep_incontext_examples(test_df, num_incontext_examples)
    prompts = []
    for i, row in test_df.iterrows():
        prompt = row["prompt"].split(", based on the following examples")[0] + ".\n\n"
        for j in incontext_indices[i]:
            ic_row = test_df.iloc[j]
            question = re.search(r".*Question:\s*(.*)", ic_row["prompt"], re.DOTALL).group(1)
            prompt += format_example(question, answer=ic_row["answer"], qa_format=qa_format) + "\n\n"

        question = re.search(r".*Question:\s*(.*)", row["prompt"], re.DOTALL).group(1)
        prompt += format_example(question, qa_format=qa_format)
        prompts.append(prompt)

    print(f"--- {split} example prompt ---\n{prompts[0]}\n----------------------")
    outputs = batched_generate(
        prompts=prompts,
        model=model,
        tokenizer=tokenizer,
        do_sample=False,
        max_new_tokens=200,
        batch_size=batch_size,
    )
    results = []
    for prompt, output, answer in zip(prompts, outputs, test_df.answer):
        #output = output.split("\n\n")[0]
        results.append(
            {
                "prompt": prompt,
                "output": output,
                "answer": answer,
                "correct": output.strip() == answer.strip(),
                "split": split,
            }
        )
    return results


@click.command()
@click.option("--model_name_or_path", type=str, default="pile-npt25k")
@click.option("--output_dir", type=str)
@click.option("--step", type=int, default=None)
@click.option("--num_incontext_examples", type=int, default=1)
@click.option("--max_num_examples", type=int, default=None)
@click.option("--eval_batch_size", type=int, default=128)
@click.option("--qa_format", type=str, default=None)
def main(
    model_name_or_path: str,
    output_dir: str,
    step: int,
    num_incontext_examples: int,
    max_num_examples: int,
    eval_batch_size: int,
    qa_format: str,
):
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, step=step)

    all_results = []
    num_splits = len(os.listdir("olmo_data/eval/cute"))
    target_tests = ["spell.jsonl", "contains_char.jsonl", "ins_char.jsonl", "del_char.jsonl", "swap_char.jsonl", "sub_char.jsonl"]
    for eval_file in os.listdir("olmo_data/eval/cute"):
        if eval_file not in target_tests:
            continue
        test_df = pd.read_json(f"olmo_data/eval/cute/{eval_file}", lines=True)
        split = eval_file.replace(".jsonl", "")
        if max_num_examples:
            test_df = test_df.sample(min(len(test_df), max_num_examples // num_splits))
        results = evaluate_cute(
            model,
            tokenizer,
            test_df,
            batch_size=eval_batch_size,
            num_incontext_examples=num_incontext_examples,
            qa_format=qa_format,
            split=split,
        )
        all_results.extend(results)
    write_results(all_results, output_dir, print_metrics=True)


if __name__ == "__main__":
    main()
