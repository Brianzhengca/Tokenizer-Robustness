import click
import pandas as pd

from eval.util import (
    batched_generate,
    format_example,
    load_model_and_tokenizer,
    prep_incontext_examples,
    write_results,
)
from olmo.util import read_json, seed_all

seed_all(42)


def evaluate_dyck_languages(model, tokenizer, test_df, batch_size, num_incontext_examples, qa_format):
    test_df = test_df.reset_index(drop=True)
    incontext_indices = prep_incontext_examples(test_df, num_incontext_examples)

    prompts = []
    for i, row in test_df.iterrows():
        prompt = "Complete the rest of the sequence, making sure that the parentheses are closed properly.\n\n"
        for j in incontext_indices[i]:
            ic_row = test_df.iloc[j]
            prompt += (
                format_example(
                    ic_row["input"],
                    answer=[k for k, v in ic_row["target_scores"].items() if v][0],
                    qa_format=qa_format,
                )
                + "\n\n"
            )
        prompt += format_example(row["input"], qa_format=qa_format)
        prompts.append(prompt)

    print(f"--- BB Dyck Languages example prompt ---\n{prompts[0]}\n----------------------")

    outputs = batched_generate(
        prompts=prompts,
        model=model,
        tokenizer=tokenizer,
        do_sample=False,
        max_new_tokens=200,
        batch_size=batch_size,
    )

    results = []
    for prompt, output, target_scores in zip(prompts, outputs, test_df.target_scores):
        #output = output.split("\n\n")[0].strip()
        answer = [k for k, v in target_scores.items() if v][0]
        results.append(
            {
                "prompt": prompt,
                "output": output,
                "answer": answer,
                "correct": output == answer,
            }
        )

    return results


@click.command()
@click.option("--model_name_or_path", type=str, default=None)
@click.option("--step", type=int, default=None)
@click.option("--output_dir", type=str, default=None)
@click.option("--num_incontext_examples", type=int, default=1)
@click.option("--max_num_examples", type=int, default=None)
@click.option("--eval_batch_size", type=int, default=32)
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
    test_df = pd.DataFrame(read_json("olmo_data/eval/bigbench-dyck-languages/task.json")["examples"])

    if max_num_examples:
        test_df = test_df.sample(min(len(test_df), max_num_examples))

    results = evaluate_dyck_languages(
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
