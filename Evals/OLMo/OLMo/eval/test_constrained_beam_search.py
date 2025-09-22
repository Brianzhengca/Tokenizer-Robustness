import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList,
)
from olmo.torch_util import seed_all
from eval.custom_logits_process import SurfaceFormConstraintLogitsProcessor

seed_all(42)

model_name_or_path = "models/hf_models/OLMo2-7B-pts200k-t180k-ctx2995/step95972"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model.eval()
constraint_beam_size = 4
max_context_length = 32

total_loss = 0
total_corrected_loss = 0
total_tokens = 0
total_bytes = 0

text = """I like pie"""


def get_segmentations(constraint):
    input_ids = tokenizer(tokenizer.eos_token, return_tensors="pt").input_ids.to(model.device)
    logits_processor = LogitsProcessorList(
        [SurfaceFormConstraintLogitsProcessor(constraint, tokenizer, verbose=True)]
    )
    model.generation_config.early_stopping = True
    outputs = model.generate(
        input_ids,
        logits_processor=logits_processor,
        num_beams=constraint_beam_size,
        num_return_sequences=constraint_beam_size,
        max_new_tokens=int(max_context_length * 1.1),  # allow found segmentation to be 10% longer
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
    )
    return outputs


inputs = tokenizer(
    tokenizer.eos_token + text + tokenizer.eos_token,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=max_context_length,
)
text = tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
print(f"Text: {text}")
input_ids = inputs.input_ids.to(model.device)

with torch.no_grad():
    # 1. calculate loss of BPE segmentation
    per_token_loss = model(input_ids=input_ids, labels=input_ids, return_dict=True).loss
    bpe_loss = (per_token_loss * (input_ids.size(1) - 1)).item()

    # 2. calculate loss of other segmentations
    segmentations = get_segmentations(constraint=tokenizer.eos_token + text)
    segmentation_ids = segmentations.sequences.to(input_ids.device)
    segmentation_scores = segmentations.sequences_scores.to(input_ids.device)

    print("------")
    print(f"BPE segmentation: {tokenizer.convert_ids_to_tokens(input_ids[0])}")
    print(f"BPE loss: {bpe_loss}")

    segmentation_lens = (segmentation_ids == tokenizer.eos_token_id).nonzero()[:, 1][1::2]
    for i in range(segmentation_ids.size(0)):
        print(f"Segmentation: {tokenizer.convert_ids_to_tokens(segmentation_ids[i])}")
        print(f"Loss: {-segmentation_scores[i].item() * segmentation_lens[i]}")

    # exclude segmentations found by beam search that are equivalent to BPE
    neq_bpe_mask = ~(segmentation_ids[:, : input_ids.size(1)] == input_ids).all(dim=-1)
    neq_bpe_mask = neq_bpe_mask.to(input_ids.device)
    segmentation_ids = segmentation_ids[neq_bpe_mask]
    segmentation_scores = segmentation_scores[neq_bpe_mask]
    print(f"segmentation_scores: {segmentation_scores}")

    # loss returned by beam search is at the token level, so we need to multiply by the length
    segmentation_lens = (segmentation_ids == tokenizer.eos_token_id).nonzero()[:, 1][1::2]
    segmentation_loss = -torch.mul(segmentation_scores, segmentation_lens)

total_loss += bpe_loss
segmentation_loss = torch.cat((segmentation_loss, torch.tensor([bpe_loss], device=segmentation_loss.device)))
total_corrected_loss += -torch.logsumexp(-torch.tensor(segmentation_loss), dim=0)

total_bytes += sum(len(text) for text in tokenizer.batch_decode(inputs.input_ids, skip_special_tokens=True))

bits_per_byte = total_loss / torch.log(torch.tensor(2.0)) / total_bytes
corrected_bits_per_byte = total_corrected_loss / torch.log(torch.tensor(2.0)) / total_bytes

print(f"Bits per byte: {bits_per_byte.item():.4f}")
print(f"Corrected bits per byte: {corrected_bits_per_byte.item():.4f}")
