import torch
from transformers import (
    AutoTokenizer,
    LogitsProcessor,
)


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
        verbose (`bool`, `optional`, defaults to `False`):
            Whether to print debug information
    """

    def __init__(self, constraint: str, tokenizer: AutoTokenizer, device: str = "cpu", verbose: bool = False):
        self.constraint = constraint
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id
        self.vocab = tokenizer.get_vocab()
        self.verbose = verbose

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.verbose:
            print("-------")

        decoded_outputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        for i, decoded_output in enumerate(decoded_outputs):
            if self.verbose:
                print(f"Seq {i}: current sequence is {self.tokenizer.convert_ids_to_tokens(input_ids[i])}")

            # when constraint is satisfied, force the next token to be eos
            if self.constraint == decoded_output:
                if self.verbose:
                    print("    Done!")
                mask = torch.zeros_like(scores[i], dtype=torch.bool)
                mask[self.eos_token_id] = True
                scores[i][~mask] = float("-inf")
                continue

            # set logits for all invalid next-tokens to -inf
            valid_next_token_ids = [
                token_id
                for token, token_id in self.vocab.items()
                if self.constraint[len(decoded_output) :].startswith(self.tokenizer.decode(token_id))
            ]
            if self.verbose:
                print(f"    Next-token choices: {self.tokenizer.convert_ids_to_tokens(valid_next_token_ids)}")
            mask = torch.zeros_like(scores[i], dtype=torch.bool)
            mask[valid_next_token_ids] = True
            scores[i][~mask] = float("-inf")
        return scores
