"""Data downloaded from https://github.com/google/BIG-bench/blob/main/bigbench/benchmark_tasks/qa_wikidata/task.json"""

import click
import pandas as pd
from eval.util import (
    load_model_and_tokenizer,
    batched_generate,
    prep_incontext_examples,
    write_results,
    format_example,
)
from olmo.util import read_json, seed_all
from tqdm import tqdm

seed_all(42)


def evaluate_wikidataqa(model, tokenizer, test_df, batch_size, num_incontext_examples, qa_format):
    test_df = test_df.reset_index(drop=True)
    incontext_indices = prep_incontext_examples(test_df, num_incontext_examples)

    prompts = []
    for i, row in tqdm(test_df.iterrows()):
        prompt = ""
        for j in incontext_indices[i]:
            incontext_row = test_df.iloc[j]
            maybe_targets = incontext_row["target"]
            target = maybe_targets if isinstance(maybe_targets, str) else maybe_targets[0]
            if qa_format == "cont":
                prompt += incontext_row["input"] + " " + target + ".\n\n"
            else:
                prompt += (
                    format_example(question=incontext_row["input"], answer=target, qa_format=qa_format) + "\n\n"
                )

        if qa_format == "cont":
            prompt += row["input"]
        else:
            prompt += format_example(question=row["input"], qa_format=qa_format)
        prompts.append(prompt)

    print(f"--- WikidataQA example prompt ---\n{prompts[0]}\n----------------------")
    outputs = batched_generate(
        prompts=prompts,
        model=model,
        tokenizer=tokenizer,
        do_sample=False,
        max_new_tokens=200,
        batch_size=batch_size,
    )
    results = []
    for prompt, output, target in zip(prompts, outputs, test_df.target):
        #output = output.split("\n\n")[0].rstrip(".,!?")
        results.append(
            {
                "prompt": prompt,
                "output": output,
                "answer": target,
                "correct": output.strip() == target if isinstance(target, str) else output.strip() in target,
            }
        )
    return results


@click.command()
@click.option("--model_name_or_path", type=str, default="pile-npt25k")
@click.option("--output_dir", type=str)
@click.option("--step", type=int, default=None)
@click.option("--max_num_examples", type=int, default=None)
@click.option("--num_incontext_examples", type=int, default=10)
@click.option("--eval_batch_size", type=int, default=128)
@click.option("--qa_format", type=str, default=None)
def main(
    model_name_or_path: str,
    output_dir: str,
    step: int,
    max_num_examples: int,
    num_incontext_examples: int,
    eval_batch_size: int,
    qa_format: str,
):
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, step=step)
    test_df = pd.DataFrame(read_json("olmo_data/eval/wikidataqa/task.json")["examples"])

    if max_num_examples:
        test_df = test_df.sample(min(len(test_df), max_num_examples))

    results = evaluate_wikidataqa(
        model,
        tokenizer,
        test_df,
        batch_size=eval_batch_size,
        num_incontext_examples=num_incontext_examples,
        qa_format=qa_format,
    )
    write_results(results, output_dir, print_metrics=True)


if __name__ == "__main__":
    main()
