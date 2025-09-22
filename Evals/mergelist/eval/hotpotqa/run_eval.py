import click
import pandas as pd
from eval.util import (
    load_model_and_tokenizer,
    batched_generate,
    format_example,
    prep_incontext_examples,
    write_results,
)
from olmo.util import read_json, seed_all

seed_all(42)


def evaluate_hotpotqa(
    model, tokenizer, test_df, batch_size, num_incontext_examples, with_passage, full_passage, qa_format
):
    test_df = test_df.reset_index(drop=True)
    incontext_indices = prep_incontext_examples(test_df, num_incontext_examples)

    def get_hotpotqa_context(row, contexts_dict):
        contexts = set()
        for title, sentence_id in row["supporting_facts"]:
            if with_passage:
                if full_passage:
                    contexts.add("".join(contexts_dict[title]))
                elif sentence_id < len(contexts_dict[title]):
                    contexts.add(contexts_dict[title][sentence_id].strip())
        return "\n".join(contexts)

    prompts = []
    for i, row in test_df.iterrows():
        prompt = ""
        for j in incontext_indices[i]:
            ic_row = test_df.iloc[j]
            contexts_dict = {k: v for k, v in ic_row["context"]}
            context = get_hotpotqa_context(ic_row, contexts_dict)
            prompt += (
                format_example(
                    ic_row["question"],
                    passage=context,
                    answer=ic_row["answer"],
                    qa_format=qa_format,
                )
                + "\n\n"
            )
        contexts_dict = {k: v for k, v in row["context"]}
        context = get_hotpotqa_context(row, contexts_dict)
        prompt += format_example(
            row["question"],
            passage=context,
            qa_format=qa_format,
        )
        prompts.append(prompt)

    print(f"--- HotpotQA example prompt ---\n{prompts[0]}\n----------------------")
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
                "correct": answer.lower() in output.lower(),
            }
        )

    return results


@click.command()
@click.option("--model_name_or_path", type=str, default="pile-npt25k")
@click.option("--step", type=int, default=None)
@click.option("--output_dir", type=str, default="results/hotpotqa/olmo-20k")
@click.option("--num_incontext_examples", type=int, default=1)
@click.option("--max_num_examples", type=int, default=None)
@click.option("--eval_batch_size", type=int, default=32)
@click.option("--with_passage", is_flag=True, default=False)
@click.option("--full_passage", is_flag=True, default=False)
@click.option("--qa_format", type=str, default="qnan")
def main(
    model_name_or_path: str,
    step: int,
    output_dir: str,
    num_incontext_examples: int,
    max_num_examples: int,
    eval_batch_size: int,
    with_passage: bool,
    full_passage: bool,
    qa_format: str,
):
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, step=step)
    test_df = pd.DataFrame(read_json("olmo_data/eval/hotpotqa/hotpot_dev_distractor_v1.json"))

    if max_num_examples:
        test_df = test_df.sample(min(len(test_df), max_num_examples))

    results = evaluate_hotpotqa(
        model,
        tokenizer,
        test_df,
        batch_size=eval_batch_size,
        num_incontext_examples=num_incontext_examples,
        with_passage=with_passage,
        full_passage=full_passage,
        qa_format=qa_format,
    )
    write_results(results, output_dir, print_metrics=True)


if __name__ == "__main__":
    main()
