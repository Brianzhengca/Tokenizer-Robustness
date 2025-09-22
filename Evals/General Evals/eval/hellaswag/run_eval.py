import click
import pandas as pd
from eval.util import (
    load_model_and_tokenizer,
    batched_generate,
    format_example,
    prep_incontext_examples,
    parse_mc_pred,
    write_results,
)
from olmo.util import seed_all

seed_all(42)


def evaluate_hellaswag(model, tokenizer, test_df, batch_size, num_incontext_examples, qa_format):
    question = "What is the best continuation?"
    test_df = test_df.reset_index(drop=True)
    incontext_indices = prep_incontext_examples(test_df, num_incontext_examples)

    prompts = []
    for i, row in test_df.iterrows():
        prompt = ""
        for j in incontext_indices[i]:
            ic_row = test_df.iloc[j]
            choices = [ic_row["ctx_b"].strip().capitalize() + " " + end for end in ic_row["endings"]]
            prompt += (
                format_example(
                    question,
                    passage=ic_row["ctx_a"].strip(),
                    choices=choices,
                    answer="ABCD"[ic_row["label"]],
                    qa_format=qa_format,
                )
                + "\n\n"
            )

        choices = [row["ctx_b"].strip().capitalize() + " " + end for end in row["endings"]]
        prompt += format_example(question, passage=row["ctx_a"].strip(), choices=choices, qa_format=qa_format)
        prompts.append(prompt)

    print(f"--- Example prompt ---\n{prompts[0]}\n----------------------")

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
    for prompt, output, label in zip(prompts, outputs, test_df.label):
        parsed_answer = parse_mc_pred(output, qa_format=qa_format)
        results.append(
            {
                "prompt": prompt,
                "output": output,
                "answer": "ABCD"[label],
                "valid": parsed_answer is not None,
                "correct": parsed_answer == "ABCD"[label],
            }
        )

    return results


@click.command()
@click.option("--model_name_or_path", type=str, default="pile-npt25k")
@click.option("--step", type=int, default=None)
@click.option("--output_dir", type=str, default="results/squad/olmo-20k")
@click.option("--max_num_examples", type=int, default=None)
@click.option("--num_incontext_examples", type=int, default=1)
@click.option("--eval_batch_size", type=int, default=32)
@click.option("--qa_format", type=str, default="qnan")
def main(
    model_name_or_path: str,
    step: int,
    output_dir: str,
    max_num_examples: int,
    eval_batch_size: int,
    num_incontext_examples: int,
    qa_format: str,
):
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, step=step)
    test_df = pd.read_json("olmo_data/eval/hellaswag/validation.jsonl", lines=True)

    if max_num_examples:
        test_df = test_df.sample(min(len(test_df), max_num_examples), random_state=42)

    results = evaluate_hellaswag(
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
