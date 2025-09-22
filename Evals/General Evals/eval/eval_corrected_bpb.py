import asyncio
import heapq
import json
import os
import uuid
from functools import partial, wraps
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm.auto as tqdm
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams, TokensPrompt

import eval.dp_tokenization as dpt
from olmo.util import bytes_to_unicode, ensure_dir


def coro(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


class CaptureAndSelectLogitsProcessor:
    def __init__(self, selections):
        self.selections = selections
        self.captured_logprobs = {}

    def __call__(self, past, logits):
        assert not past
        logprobs = F.log_softmax(logits, 0)
        mask = torch.ones_like(logits) * -torch.inf
        for tid in self.selections:
            mask[tid] = 0
            self.captured_logprobs[tid] = logprobs[tid].cpu().item()

        return logits + mask


class ByteBeamSearch:
    def __init__(self):
        self.btu = bytes_to_unicode()
        self.utb = {v: k for k, v in self.btu.items()}

    @classmethod
    async def create(cls, llm):
        self = cls()
        self.llm = llm
        self.tokenizer = await llm.get_tokenizer()
        self.vocab = {bytes(self.utb[c] for c in tok): tid for tok, tid in self.tokenizer.vocab.items()}
        self.vrev = {tid: tok for tok, tid in self.vocab.items()}
        self.vtrie = dpt.build_trie(self.vocab.keys())
        self.rtrie = dpt.build_trie(map(reversed, self.vocab.keys()))
        for tok, tid in self.vocab.items():
            pointer = self.vtrie
            for b in tok:
                pointer = pointer[b]
            pointer[None] = tid

            rpointer = self.rtrie
            for b in reversed(tok):
                rpointer = rpointer[b]
            rpointer[None] = tid

        return self

    async def generate(self, *args, **kwargs):
        request_id = uuid.uuid4()
        outputs = self.llm.generate(*args, **kwargs, request_id=request_id)
        async for output in outputs:
            final_output = output
        return final_output

    async def get_logprobs(self, tokens, selections):
        cslp = CaptureAndSelectLogitsProcessor(selections)
        result = await self.generate(
            TokensPrompt(prompt_token_ids=tokens),
            SamplingParams(
                max_tokens=1,
                temperature=0,
                logits_processors=[cslp],
            ),
        )
        return result, cslp.captured_logprobs

    async def reference_score(self, tokens, add_leading_eos=True, add_trailing_eos=True):
        lps = (
            await self.generate(
                TokensPrompt(
                    prompt_token_ids=([self.tokenizer.eos_token_id] if add_leading_eos else [])
                    + tokens
                    + ([self.tokenizer.eos_token_id] if add_trailing_eos else [])
                ),
                SamplingParams(prompt_logprobs=0, max_tokens=1),
            )
        ).prompt_logprobs
        return sum(next(iter(lp.values())).logprob for lp in lps if lp is not None)

    async def byte_beam_search(self, text: bytes, beam_width=3, progress=False, score_eos=True):
        assert isinstance(text, bytes)
        eos = self.tokenizer.eos_token_id
        table = [[[0.0, eos, None, None]]]

        def get_prompt(beam):
            # walk the beam backwards to get the prompt
            bpointer, prompt_rev = beam, []
            while True:
                prompt_rev.append(bpointer[1])
                if bpointer[2] is None:
                    break
                bpointer = table[bpointer[2][0]][bpointer[2][1]]

            return prompt_rev[::-1]

        async def fetch_logprobs(j, beam):
            if beam[3] is None:
                prompt = get_prompt(beam)

                # scan forwards to get valid tokens
                valid_tokens, pointer = [], self.vtrie
                for k in range(j, len(text)):
                    if (pointer := pointer.get(text[k])) is None:
                        # print(f"break forward scan {k, text[j]}")
                        break
                    if (tid := pointer.get(None)) is not None:
                        valid_tokens.append(tid)

                _, logprobs = await self.get_logprobs(prompt, valid_tokens)
                beam[3] = logprobs

            return beam[3]

        R = partial(tqdm.trange, position=tqdm.tqdm._get_free_pos(), leave=False) if progress else range
        for i in R(1, len(text) + 1):
            candidates = []

            # scan backward to get valid tokens
            rpointer = self.rtrie
            for j in range(i - 1, -1, -1):
                if (rpointer := rpointer.get(text[j])) is None:
                    # print(f"break reverse scan {j, text[j]}")
                    break
                if (tid := rpointer.get(None)) is not None:
                    for k, beam in enumerate(table[j]):
                        logprobs = await fetch_logprobs(j, beam)
                        if (logprob := logprobs.get(tid)) is not None:
                            cum_logprob = logprob + beam[0]
                            candidate = [cum_logprob, tid, (j, k), None]
                            candidates.append(candidate)
                        else:
                            print(f"warning: pruning beam {(j, k)} due to context limit")

            if not candidates:
                print("warning: no beams fit in context limit")
                return

            cur_width = beam_width(i) if callable(beam_width) else beam_width
            new_beams = heapq.nlargest(cur_width, candidates)
            table.append(new_beams)

        final_beams, result = table[-1], {}
        for k, beam in enumerate(final_beams):
            prompt, score = get_prompt(beam), beam[0]

            # To taste: score terminating EOS
            if score_eos:
                _, logprobs = await self.get_logprobs(prompt, [eos])
                if eos not in logprobs:
                    print(f"warning: pruning beam {(len(table), k)} due to context limit")
                    continue

                score += logprobs[eos]

            # To taste: remove the initial EOS token
            result[tuple(prompt[1:])] = score

        return result


def beam_schedule(text_len, beam_min=2, beam_max=1000):
    lmin, lmax = np.log(beam_min), np.log(beam_max)

    def inner(i):
        return int(np.ceil(np.exp(lmin + (lmax - lmin) * (i - 1) / (text_len - 1))))

    return inner


async def run_document(BBS, i, text, max_ctx, dest_file, beam_min, beam_max, verbose=False):
    eos = BBS.tokenizer.eos_token_id
    tokens = BBS.tokenizer.encode(text)
    tokens.append(eos)
    tokens = tokens[:max_ctx]
    score_eos = False
    tokens_clipped = tokens
    if tokens[-1] == eos:
        score_eos = True
        tokens_clipped = tokens[:-1]

    text_clipped = BBS.tokenizer.decode(tokens_clipped).encode()
    ref_loss = -await BBS.reference_score(tokens_clipped)
    ref_bpb = ref_loss / len(text_clipped) / np.log(2)

    result = await BBS.byte_beam_search(
        text_clipped,
        beam_width=beam_schedule(len(text_clipped), beam_min, beam_max),
        progress=True,
        score_eos=score_eos,
    )

    bpe_in_beams = False
    our_losses = [ref_loss]
    for token_ids, score in result.items():
        if token_ids == tokens_clipped:
            bpe_in_beams = True
            continue
        our_losses.append(-score)

    if verbose:
        print(f"BPE segmentation: {BBS.tokenizer.convert_ids_to_tokens(tokens_clipped)}")
        print(f"Reference loss: {ref_loss}")

        for token_ids, score in list(result.items())[:5]:
            print(f"Segmentation: {BBS.tokenizer.convert_ids_to_tokens(token_ids)}")
            print(f"Loss: {score}")

    corrected_loss = -torch.logsumexp(-torch.tensor(our_losses), 0).item()
    corrected_bpb = corrected_loss / len(text_clipped) / np.log(2)
    beams_loss = -torch.logsumexp(torch.tensor(list(result.values())), 0).item()
    beams_bpb = beams_loss / len(text_clipped) / np.log(2)

    tqdm.tqdm.write(f"{i}: {ref_bpb:.6f}, {corrected_bpb:.6f}, {ref_bpb - corrected_bpb:.6f}")
    with open(dest_file, "wt") as fout:
        output = {
            "bpe": (tokens_clipped, ref_loss),
            "beams": list(result.items()),
            "ref_loss": ref_loss,
            "ref_bpb": ref_bpb,
            "corrected_loss": corrected_loss,
            "corrected_bpb": corrected_bpb,
            "beams_loss": beams_loss,
            "beams_bpb": beams_bpb,
            "len": len(text_clipped),
            "bpe_in_beams": str(bpe_in_beams).lower(),
        }
        json.dump(output, fout)


@click.command()
@click.option("--data_dir", default="olmo_data/olmo2_shuffle")
@click.option("--model_name_or_path", default="models/hf_models/OLMo2-7B-pts200k-t180k-ctx2995/step95972")
@click.option("--output_dir", default="results/dp_tcs_bpb/OLMo2-7B-pts200k-t180k-ctx2995/step95972")
@click.option("--max_num_examples", default=1)
@click.option("--max_context_length", default=128)
@click.option("--batch_size", default=32)
@click.option("--beam_min", default=2)
@click.option("--beam_max", default=100)
@coro
async def main(
    data_dir: str,
    model_name_or_path: str,
    output_dir: str,
    max_num_examples: int,
    max_context_length: int,
    batch_size: int,
    beam_min: int,
    beam_max: int,
):
    data_dir = Path(data_dir)
    model_dir, output_dir = Path(model_name_or_path), Path(output_dir)
    ensure_dir(output_dir)
    ensure_dir(output_dir / "data")
    llm = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            str(model_dir),
            enforce_eager=True,
            disable_log_stats=True,
            enable_chunked_prefill=True,
            max_num_batched_tokens=512,
            enable_prefix_caching=True,
        ),
    )
    llm.log_requests = False
    BBS = await ByteBeamSearch.create(llm)

    # read .jsonl files from eval_dir until we have max_num_examples
    data = []
    for eval_file in os.listdir(data_dir):
        nrows = max_num_examples - len(data)
        df = pd.read_json(data_dir / eval_file, lines=True, compression="zstd", nrows=nrows)
        data.extend(df.text.tolist())
        if len(data) >= max_num_examples:
            break

    dltasks = set()
    for i in range(max_num_examples):
        dest_file = output_dir / f"data/{i}.json"
        if os.path.exists(dest_file):
            continue
        if len(dltasks) >= batch_size:
            _done, dltasks = await asyncio.wait(dltasks, return_when=asyncio.FIRST_COMPLETED)
        dltasks.add(
            asyncio.create_task(run_document(BBS, i, data[i], max_context_length, dest_file, beam_min, beam_max))
        )
    await asyncio.wait(dltasks)

    # read results
    total_loss, total_corrected_loss, total_beams_loss, total_bytes = 0, 0, 0
    for i in range(max_num_examples):
        with open(output_dir / f"data/{i}.json", "rt") as fin:
            result = json.load(fin)
            total_loss += result["ref_loss"]
            total_corrected_loss += result["corrected_loss"]
            total_beams_loss += result["beams_loss"]
            total_bytes += result["len"]

    metrics = {
        "bpb": total_loss / total_bytes / np.log(2),
        "corrected_bpb": total_corrected_loss / total_bytes / np.log(2),
        "beams_bpb": total_beams_loss / total_bytes / np.log(2),
        "total_bytes": total_bytes,
    }

    with open(output_dir / "metrics.json", "wt") as fout:
        json.dump(metrics, fout)


if __name__ == "__main__":
    main()
