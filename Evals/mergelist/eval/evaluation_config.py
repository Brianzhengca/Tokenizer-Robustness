from eval.arc.run_eval import evaluate_arc
from eval.boolq.run_eval import evaluate_boolq
from eval.copa.run_eval import evaluate_copa
from eval.coqa.run_eval import evaluate_coqa
from eval.drop.run_eval import evaluate_drop
from eval.hellaswag.run_eval import evaluate_hellaswag
from eval.hotpotqa.run_eval import evaluate_hotpotqa
from eval.jeopardy.run_eval import evaluate_jeopardy
from eval.lambada.run_eval import evaluate_lambada
from eval.mmlu.run_eval import evaluate_mmlu
from eval.mmlu_pro.run_eval import evaluate_mmlu_pro
from eval.openbookqa.run_eval import evaluate_openbookqa
from eval.squad.run_eval import evaluate_squad
from eval.tofu.run_eval import evaluate_tofu
from eval.triviaqa.run_eval import evaluate_triviaqa
from eval.wikidataqa.run_eval import evaluate_wikidataqa
from eval.winogrande.run_eval import evaluate_winogrande

NUM_INCONTEXT_EXAMPLES = 5
MAX_NUM_EXAMPLES = 1000
QA_FORMAT = "q"


EVALUATION_CONFIGS = {
    "ARC-Easy": {
        "path": "olmo_data/eval/arc-easy/test.jsonl",
        "eval_func": evaluate_arc,
        "max_num_examples": MAX_NUM_EXAMPLES,
        "num_incontext_examples": NUM_INCONTEXT_EXAMPLES,
        "batch_size": 64,
        "qa_format": QA_FORMAT,
    },
    "ARC-Challenge": {
        "path": "olmo_data/eval/arc-challenge/test.jsonl",
        "eval_func": evaluate_arc,
        "max_num_examples": MAX_NUM_EXAMPLES,
        "num_incontext_examples": NUM_INCONTEXT_EXAMPLES,
        "batch_size": 64,
        "qa_format": QA_FORMAT,
    },
    "BoolQ": {
        "path": "olmo_data/eval/boolq/validation.jsonl",
        "eval_func": evaluate_boolq,
        "max_num_examples": MAX_NUM_EXAMPLES,
        "num_incontext_examples": NUM_INCONTEXT_EXAMPLES,
        "batch_size": 32,
        "qa_format": QA_FORMAT,
    },
    "COPA": {
        "path": "olmo_data/eval/copa/test.jsonl",
        "eval_func": evaluate_copa,
        "max_num_examples": MAX_NUM_EXAMPLES,
        "num_incontext_examples": NUM_INCONTEXT_EXAMPLES,
        "batch_size": 64,
        "qa_format": QA_FORMAT,
    },
    "CoQA": {
        "path": "olmo_data/eval/coqa/coqa-dev-v1.0.json",
        "eval_func": evaluate_coqa,
        "max_num_examples": 300,
        "batch_size": 32,
        "qa_format": QA_FORMAT,
    },
    "DROP": {
        "path": "olmo_data/eval/drop/validation.jsonl",
        "eval_func": evaluate_drop,
        "max_num_examples": MAX_NUM_EXAMPLES,
        "num_incontext_examples": NUM_INCONTEXT_EXAMPLES,
        "batch_size": 16,
        "qa_format": QA_FORMAT,
    },
    "HellaSwag": {
        "path": "olmo_data/eval/hellaswag/validation.jsonl",
        "eval_func": evaluate_hellaswag,
        "max_num_examples": MAX_NUM_EXAMPLES,
        "num_incontext_examples": NUM_INCONTEXT_EXAMPLES,
        "batch_size": 32,
        "qa_format": QA_FORMAT,
    },
    "HotpotQA": {
        "path": "olmo_data/eval/hotpotqa/hotpot_dev_distractor_v1.json",
        "eval_func": evaluate_hotpotqa,
        "max_num_examples": MAX_NUM_EXAMPLES,
        "num_incontext_examples": NUM_INCONTEXT_EXAMPLES,
        "batch_size": 32,
        "with_passage": True,
        "full_passage": True,
        "qa_format": QA_FORMAT,
    },
    "Jeopardy": {
        "path": "olmo_data/eval/jeopardy/test.jsonl",
        "eval_func": evaluate_jeopardy,
        "max_num_examples": MAX_NUM_EXAMPLES,
        "num_incontext_examples": NUM_INCONTEXT_EXAMPLES,
        "batch_size": 128,
        "qa_format": QA_FORMAT,
    },
    "Lambada": {
        "path": "olmo_data/eval/lambada/test.jsonl",
        "eval_func": evaluate_lambada,
        "max_num_examples": MAX_NUM_EXAMPLES,
        "num_incontext_examples": NUM_INCONTEXT_EXAMPLES,
        "batch_size": 64,
        "qa_format": QA_FORMAT,
    },
    "MMLU": {
        "path": "olmo_data/eval/mmlu/test.jsonl",
        "eval_func": evaluate_mmlu,
        "max_num_examples": MAX_NUM_EXAMPLES,
        "num_incontext_examples": NUM_INCONTEXT_EXAMPLES,
        "batch_size": 32,
        "qa_format": QA_FORMAT,
    },
    "MMLU-Pro": {
        "path": "olmo_data/eval/mmlu-pro/test.jsonl",
        "eval_func": evaluate_mmlu_pro,
        "max_num_examples": MAX_NUM_EXAMPLES,
        "num_incontext_examples": NUM_INCONTEXT_EXAMPLES,
        "batch_size": 32,
        "qa_format": QA_FORMAT,
    },
    "OpenbookQA": {
        "path": "olmo_data/eval/openbookqa/test.jsonl",
        "eval_func": evaluate_openbookqa,
        "max_num_examples": MAX_NUM_EXAMPLES,
        "num_incontext_examples": NUM_INCONTEXT_EXAMPLES,
        "batch_size": 64,
        "qa_format": QA_FORMAT,
    },
    "SQuAD": {
        "path": "olmo_data/eval/squad/validation.jsonl",
        "eval_func": evaluate_squad,
        "max_num_examples": MAX_NUM_EXAMPLES,
        "num_incontext_examples": NUM_INCONTEXT_EXAMPLES,
        "batch_size": 32,
        "qa_format": QA_FORMAT,
    },
    "TOFU": {
        "path": "olmo_data/eval/tofu/world_facts.jsonl",
        "eval_func": evaluate_tofu,
        "max_num_examples": MAX_NUM_EXAMPLES,
        "num_incontext_examples": NUM_INCONTEXT_EXAMPLES,
        "batch_size": 128,
        "qa_format": QA_FORMAT,
    },
    "TriviaQA": {
        "path": "olmo_data/eval/triviaqa/validation.jsonl",
        "eval_func": evaluate_triviaqa,
        "max_num_examples": MAX_NUM_EXAMPLES,
        "num_incontext_examples": NUM_INCONTEXT_EXAMPLES,
        "batch_size": 128,
        "qa_format": QA_FORMAT,
    },
    "WikidataQA": {
        "path": "olmo_data/eval/wikidataqa/task.json",
        "eval_func": evaluate_wikidataqa,
        "max_num_examples": MAX_NUM_EXAMPLES,
        "num_incontext_examples": NUM_INCONTEXT_EXAMPLES,
        "batch_size": 128,
        "qa_format": "cont",
    },
    "Winogrande": {
        "path": "olmo_data/eval/winogrande/dev.jsonl",
        "eval_func": evaluate_winogrande,
        "max_num_examples": MAX_NUM_EXAMPLES,
        "num_incontext_examples": NUM_INCONTEXT_EXAMPLES,
        "batch_size": 128,
        "qa_format": QA_FORMAT,
    },
}
