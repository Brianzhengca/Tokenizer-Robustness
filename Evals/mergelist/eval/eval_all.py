import os

import click
import pandas as pd
from eval.util import load_model_and_tokenizer, write_results
from eval.evaluation_config import EVALUATION_CONFIGS
from olmo.util import read_json, seed_all

seed_all(42)


@click.command()
@click.option("--model_name_or_path", type=str, default="pile-npt25k")
@click.option("--step", type=int, default=None)
@click.option("--overwrite", is_flag=True, default=False)
@click.option("--qa_format", type=str, default=None)
def main(model_name_or_path: str, step: int, overwrite: bool, qa_format: str):
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, step=step)

    for task_name in EVALUATION_CONFIGS:
        model_name = os.path.basename(model_name_or_path)
        task_config = EVALUATION_CONFIGS[task_name]
        kwargs = {
            key: task_config[key] for key in task_config if key not in {"path", "eval_func", "max_num_examples"}
        }
        if qa_format:  # override qa_format if provided
            kwargs["qa_format"] = qa_format

        output_dir = f"results/{task_name.lower()}"
        if kwargs["qa_format"] != "qnan":
            output_dir += f"-{kwargs['qa_format']}"
        if "num_incontext_examples" in kwargs:
            output_dir += f"-ice{kwargs['num_incontext_examples']}"
        output_dir += f"/{model_name}/{step}"

        if not os.path.exists(output_dir) or overwrite:
            test_path = EVALUATION_CONFIGS[task_name]["path"]
            if test_path.endswith(".jsonl"):
                test_df = pd.read_json(test_path, lines=True)
            elif test_path.endswith(".json"):
                test_data = read_json(test_path)
                for field in ["data", "examples"]:
                    if field in test_data:
                        test_data = test_data[field]
                        break
                test_df = pd.DataFrame(test_data)

            if task_config["max_num_examples"]:
                test_df = test_df.sample(min(len(test_df), task_config["max_num_examples"]), random_state=42)
            results = EVALUATION_CONFIGS[task_name]["eval_func"](model, tokenizer, test_df, **kwargs)
            write_results(results, output_dir)


if __name__ == "__main__":
    main()
