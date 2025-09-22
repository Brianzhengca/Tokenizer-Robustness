import re
import click
import pandas as pd
from eval.util import (
    load_model_and_tokenizer,
    batched_generate,
    format_example,
    prep_incontext_examples,
    write_results,
    parse_number,
)
from olmo.util import seed_all

seed_all(42)


def evaluate_gsm(model, tokenizer, test_df, batch_size, num_incontext_examples, qa_format):
    test_df = test_df.reset_index(drop=True)
    incontext_indices = prep_incontext_examples(test_df, num_incontext_examples)

    def format_answer(answer):
        answer = re.sub(r"<<.*?>>", "", answer)
        final_answer = answer.split("####")[-1].strip()
        sentences = answer.split("####")[0].strip().split("\n")
        sentences = [s + "." if not s.endswith(".") else s for s in sentences]
        return " ".join(sentences) + f"\n#### {final_answer}"

    prompts = []
    for i, row in test_df.iterrows():
        prompt = ""
        for j in incontext_indices[i]:
            ic_row = test_df.iloc[j]
            prompt += (
                format_example(ic_row["question"], answer=format_answer(ic_row["answer"]), qa_format=qa_format)
                + "\n\n"
            )

        prompt += format_example(row["question"], qa_format=qa_format)
        prompts.append(prompt)

    print(f"--- GSM8K example prompt ---\n{prompts[0]}\n----------------------")
    outputs = batched_generate(
        prompts=prompts,
        model=model,
        tokenizer=tokenizer,
        do_sample=False,
        max_new_tokens=512,
        batch_size=batch_size,
    )
    results = []
    for prompt, output, answer in zip(prompts, outputs, test_df.answer):
        #output = output.split("\n\n")[0]
        pred = parse_number(output.split("####")[-1])
        short_answer = parse_number(answer.split("####")[-1])
        results.append(
            {
                "prompt": prompt,
                "output": output,
                "answer": answer,
                "correct": pred == short_answer,
            }
        )
    return results


@click.command()
@click.option("--model_name_or_path", type=str, default="pile-npt25k")
@click.option("--output_dir", type=str)
@click.option("--step", type=int, default=None)
@click.option("--num_incontext_examples", type=int, default=1)
@click.option("--max_num_examples", type=int, default=None)
@click.option("--eval_batch_size", type=int, default=64)
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
    test_df = pd.read_json("olmo_data/eval/gsm8k/test.jsonl", lines=True)

    if max_num_examples:
        test_df = test_df.sample(min(len(test_df), max_num_examples))

    results = evaluate_gsm(
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
