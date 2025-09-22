"""Download data from https://github.com/ybisk/ybisk.github.io/tree/master/piqa/data"""

import click
import pandas as pd
from eval.util import (
    load_model_and_tokenizer,
    batched_generate,
    format_example,
    parse_mc_pred,
    prep_incontext_examples,
    write_results,
)
from olmo.util import seed_all

seed_all(42)


def evaluate_piqa(model, tokenizer, test_df, batch_size, num_incontext_examples, qa_format="qnan"):
    test_df = test_df.reset_index(drop=True)
    incontext_indices = prep_incontext_examples(test_df, num_incontext_examples)

    prompts = []
    for i, row in test_df.iterrows():
        prompt = ""
        for j in incontext_indices[i]:
            ic_row = test_df.iloc[j]
            options = [ic_row["sol1"], ic_row["sol2"]]
            prompt += (
                format_example(ic_row["goal"], choices=options, answer="AB"[ic_row["answer"]], qa_format=qa_format)
                + "\n\n"
            )

        options = [row["sol1"], row["sol2"]]
        prompt += format_example(row["goal"], choices=options, qa_format=qa_format)
        prompts.append(prompt)

    print(f"--- PIQA example prompt ---\n{prompts[0]}\n----------------------")

    outputs = batched_generate(
        prompts=prompts,
        model=model,
        tokenizer=tokenizer,
        is_mcq=True,
        do_sample=False,
        max_new_tokens=200,
        batch_size=batch_size,
    )

    results = []
    for prompt, output, answer in zip(prompts, outputs, test_df["answer"]):
        parsed_pred = parse_mc_pred(output, num_options=2, qa_format=qa_format)
        results.append(
            {
                "prompt": prompt,
                "output": output,
                "answer": "AB"[answer],
                "valid": parsed_pred is not None,
                "correct": parsed_pred == "AB"[answer],
            }
        )

    return results


@click.command()
@click.option("--model_name_or_path", type=str, default="pile-npt25k")
@click.option("--step", type=int, default=None)
@click.option("--output_dir", type=str, default="results/squad/olmo-20k")
@click.option("--num_incontext_examples", type=int, default=1)
@click.option("--max_num_examples", type=int, default=None)
@click.option("--eval_batch_size", type=int, default=64)
@click.option("--qa_format", type=str, default=None)
def main(
    model_name_or_path: str,
    step: int,
    output_dir: str,
    num_incontext_examples: int,
    max_num_examples: int,
    eval_batch_size: int,
    qa_format: str,
):
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, step=step)
    test_df = pd.read_json("olmo_data/eval/piqa/valid.jsonl", lines=True)
    answers = open("olmo_data/eval/piqa/valid-labels.lst").read().splitlines()
    test_df["answer"] = [int(a) for a in answers]

    if max_num_examples:
        test_df = test_df.sample(min(len(test_df), max_num_examples), random_state=42)

    results = evaluate_piqa(
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
