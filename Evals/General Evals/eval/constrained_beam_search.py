import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)
from olmo.torch_util import seed_all

seed_all(42)

model_name_or_path = "models/hf_models/OLMo2-7B-pts200k-t180k-ctx2995/step95972"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


class SurfaceFormConstraintLogitsProcessor(LogitsProcessor):
    """
    [`LogitsProcessor`] which forces the generated sequence of tokens to decode to a specific surface form.

    Args:
        constraint (`str`):
            The surface form that the model should generate
        eos_token_id (`Union[int, List[int], torch.Tensor]`):
            The id(s) of the *end-of-sequence* token.
        device (`str`, *optional*, defaults to `"cpu"`):
            The device to allocate the tensors.
    """

    def __init__(self, constraint: str, tokenizer: AutoTokenizer, device: str = "cpu"):
        self.constraint = constraint
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id
        self.vocab = tokenizer.get_vocab()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        decoded_outputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        for i, decoded_output in enumerate(decoded_outputs):
            print(f"Seq {i}: current sequence is {self.tokenizer.convert_ids_to_tokens(input_ids[i])}")

            # when constraint is satisfied, force the next token to be eos
            if self.constraint == decoded_output:
                print("    Done!")
                mask = torch.zeros_like(scores[i], dtype=torch.bool)
                mask[self.eos_token_id] = True
                scores[i][~mask] = float("-inf")
                continue

            # set logits for all invalid next-tokens to -inf
            valid_next_token_ids = [
                token_id
                for token, token_id in self.vocab.items()
                if self.constraint.replace(decoded_output, "").startswith(self.tokenizer.decode(token_id))
            ]
            print(f"    Valid next-token choices are {self.tokenizer.convert_ids_to_tokens(valid_next_token_ids)}")
            mask = torch.zeros_like(scores[i], dtype=torch.bool)
            mask[valid_next_token_ids] = True
            scores[i][~mask] = float("-inf")
        return scores


input_text = tokenizer.eos_token
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
print(input_ids)

# constraint = "Lexical analysis is the conversion of a text into meaningful lexical tokens based on a lexical grammar. Learn about the stages, categories, and examples of lexical tokens, and the difference between lexical analysis and large language models."
constraint = [
    "Lexical analysis is the conversion of a text into meaningful lexical tokens based on a lexical grammar.",
    "Hello world!",
]
logits_processor = LogitsProcessorList([SurfaceFormConstraintLogitsProcessor(constraint, tokenizer.eos_token_id)])
model.generation_config.early_stopping = True
outputs = model.generate(
    input_ids,
    logits_processor=logits_processor,
    num_beams=4,
    num_return_sequences=4,
    max_new_tokens=100,
    do_sample=False,
    output_scores=True,
    return_dict_in_generate=True,
)

all_losses = []
num_text_bytes = 0

print("----")
bpe_ids = tokenizer(tokenizer.eos_token + constraint, return_tensors="pt").input_ids[0]
print(f"bpe_ids: {bpe_ids}")
print(tokenizer.convert_ids_to_tokens(bpe_ids))
canonical_loss = model(bpe_ids.unsqueeze(0), labels=bpe_ids.unsqueeze(0)).loss * bpe_ids.size(0)
all_losses.append(canonical_loss)
num_text_bytes += len(constraint)
print(f"Sequence loss: {canonical_loss}")

for o_ids in outputs.sequences:
    print("----")
    indices = (o_ids == tokenizer.eos_token_id).nonzero().squeeze()
    start_idx, end_idx = indices[0], indices[1]
    o_ids = o_ids[start_idx:end_idx]
    print(f"output_ids: {o_ids}")
    print(tokenizer.convert_ids_to_tokens(o_ids))
    if torch.equal(o_ids, bpe_ids):
        print("Equivalent to BPE segmentation")
        continue
    per_token_loss = model(o_ids.unsqueeze(0), labels=o_ids.unsqueeze(0)).loss
    seq_loss = per_token_loss * o_ids.size(0)
    all_losses.append(seq_loss)
    num_text_bytes += len(constraint)
    print(f"Sequence loss: {seq_loss}")

print("----")
original_bpb = (canonical_loss / torch.log(torch.tensor(2.0)) / len(constraint)).item()
modified_bpb = (
    -torch.logsumexp(-torch.tensor(all_losses), dim=0) / torch.log(torch.tensor(2.0)) / len(constraint)
).item()
print(f"original_bpb: {original_bpb}")
print(f"modified_bpb: {modified_bpb}")
