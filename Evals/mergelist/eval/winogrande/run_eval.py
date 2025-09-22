"""Data downloaded from https://storage.googleapis.com/ai2-mosaic/public/winogrande/winogrande_1.1.zip"""

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


def evaluate_winogrande(model, tokenizer, test_df, batch_size, num_incontext_examples, qa_format="qnan"):
    test_df = test_df.reset_index(drop=True)
    incontext_indices = prep_incontext_examples(test_df, num_incontext_examples)

    prompts = []
    for i, row in test_df.iterrows():
        prompt = ""
        for j in incontext_indices[i]:
            ic_row = test_df.iloc[j]
            question = ic_row["sentence"].replace("_", "___")
            options = [ic_row["option1"], ic_row["option2"]]
            prompt += (
                format_example(
                    question,
                    choices=options,
                    answer="AB"[ic_row["answer"] - 1],
                    qa_format=qa_format,
                    question_prefix="Fill in the blank:",
                )
                + "\n\n"
            )

        question = row["sentence"].replace("_", "___")
        options = [row["option1"], row["option2"]]
        prompt += format_example(
            question, choices=options, qa_format=qa_format, question_prefix="Fill in the blank:"
        )
        prompts.append(prompt)

    print(f"--- Winogrande example prompt ---\n{prompts[0]}\n----------------------")

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
                "answer": "AB"[answer - 1],
                "valid": parsed_pred is not None,
                "correct": parsed_pred == "AB"[answer - 1],
            }
        )

    return results


@click.command()
@click.option("--model_name_or_path", type=str, default=None)
@click.option("--step", type=int, default=None)
@click.option("--output_dir", type=str, default=None)
@click.option("--num_incontext_examples", type=int, default=1)
@click.option("--max_num_examples", type=int, default=None)
@click.option("--eval_batch_size", type=int, default=128)
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
    test_df = pd.read_json("olmo_data/eval/winogrande/dev.jsonl", lines=True)

    if max_num_examples:
        test_df = test_df.sample(min(len(test_df), max_num_examples), random_state=42)

    results = evaluate_winogrande(
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
