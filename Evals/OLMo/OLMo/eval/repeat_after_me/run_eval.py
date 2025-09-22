import click
import pandas as pd
import json
import numpy as np
from pathlib import Path
from eval.util import load_model_and_tokenizer, batched_generate
from olmo.util import ensure_dir, seed_all
import string

seed_all(42)


def construct_test_data(tokenizer):
    def maybe_english(token):
        return all([c in string.ascii_letters for c in token.replace("Ġ", "")])

    def is_multiword(token):
        return token.startswith("Ġ") and token.count("Ġ") > 1 and token.count("ĠĠ") == 0  # not code

    # collect multi-word tokens
    vocab = tokenizer.get_vocab()
    multiword_tokens = []
    for token in vocab:
        if maybe_english(token) and is_multiword(token):
            multiword_tokens.append(token)

    # construct test examples
    test_examples = []
    for token in multiword_tokens:
        words = token.split("Ġ")[1:]
        prompt = f"Repeat after me: {(' '.join(words) + ', ') * 100}{words[0]}"
        answer = " " + " ".join(words[1:])
        test_examples.append({"prompt": prompt, "token": token, "answer": answer})

    # print some examples
    for i in range(5):
        print(test_examples[i]["prompt"])

    return pd.DataFrame(test_examples)


def evaluate_repeat_after_me(model, tokenizer, test_df, eval_batch_size):
    prompts = test_df.prompt.tolist()

    kwargs = {
        "max_new_tokens": 200,
        "do_sample": False,
        "batch_size": eval_batch_size,
    }
    results = {}
    outputs = batched_generate(prompts, model, tokenizer, **kwargs)

    def score(answers, outputs):
        correct = []
        for answer, output in zip(answers, outputs):
            correct.append(output.startswith(answer.replace("Ġ", " ")))
        return correct

    correct = score(test_df.answer.tolist(), outputs)
    results["original"] = [{"prompt": p, "output": o, "correct": c} for p, o, c in zip(prompts, outputs, correct)]

    # calculate accuracy with manual backoff, which represents the performance upper bound
    backed_off_prompts = [p.rsplit(" ", 1)[0] for p in prompts]  # back off one word
    outputs = batched_generate(backed_off_prompts, model, tokenizer, **kwargs)
    correct = score(test_df.token.tolist(), outputs)
    results["backed_off"] = [
        {"prompt": p, "output": o, "correct": c} for p, o, c in zip(backed_off_prompts, outputs, correct)
    ]

    return results


@click.command()
@click.option("--model_name_or_path", type=str, default="pile-npt25k")
@click.option("--step", type=int, default=None)
@click.option("--output_dir", type=str, default="results/bpb/olmo-20k")
@click.option("--max_num_examples", type=int, default=None)
@click.option("--eval_batch_size", type=int, default=64)
@click.option("--add_bos_token", is_flag=True, default=False)
def main(
    model_name_or_path: str,
    step: int,
    output_dir: str,
    max_num_examples: int,
    eval_batch_size: int,
    add_bos_token: bool,
):
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, step=step, add_bos_token=add_bos_token)
    test_df = construct_test_data(tokenizer)

    if max_num_examples:
        test_df = test_df.sample(min(len(test_df), max_num_examples))

    results = evaluate_repeat_after_me(model, tokenizer, test_df, eval_batch_size)
    metrics = {
        "accuracy": np.mean([r["correct"] for r in results["original"]]),
        "backed_off_accuracy": np.mean([r["correct"] for r in results["backed_off"]]),
        "num_examples": len(results["original"]),
    }

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

    for key, result in results.items():
        filename = "predictions.jsonl" if key == "original" else f"predictions_{key}.jsonl"
        pd.DataFrame(result).to_json(output_dir / filename, orient="records", lines=True)

    test_df.to_json(output_dir / "test_data.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    main()
