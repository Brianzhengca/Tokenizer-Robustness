from pathlib import Path
import os
import numpy as np
import random
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import base64
from collections import defaultdict
from ahocorasick_rs import BytesAhoCorasick
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

def check_valid_token(token):
    return all(c in r"QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm" for c in token)
def encode_filename(token):
    # Encode as bytes first:
    token_bytes = token.encode("utf-8")
    # Base64 encode:
    encoded = base64.urlsafe_b64encode(token_bytes).decode("utf-8")
    # This ensures only [A-Za-z0-9_-] characters are used.
    return encoded

raw_vocab = tokenizer.get_vocab()
vocab = [token for token in raw_vocab if check_valid_token(token)]

def search_tokens(tokens_of_interest):
    # Build Aho-Corasick automaton for multiple tokens
    patterns = [t.encode("utf-8") for t in tokens_of_interest]
    ac = BytesAhoCorasick(patterns)

    corpus_dir = Path("path_to_oscar")
    lang = "en"
    lang_files = os.listdir(corpus_dir / lang)

    # Dictionary of dictionaries:
    # contexts_per_token = {token: {context_str: count, ...}, ...}
    contexts_per_token = {t: defaultdict(int) for t in tokens_of_interest}

    for fname in tqdm(lang_files):
        orig_path = corpus_dir / lang / fname
        os.system(f'cp {orig_path} {Path("/scr")}')

        with open(f"/scr/{fname}", "rb") as f:
            text = f.read()

        # Find all matches for all tokens in one pass
        matches = ac.find_matches_as_indexes(text, overlapping=True)
        # matches: List of (pattern_index, start, end)

        # Process each match
        for pattern_index, start, end in matches:
            token_of_interest = tokens_of_interest[pattern_index]

            # Get the whitespace-delimited word that contains the token
            wstart, wend = start, end
            while wstart > 0 and text[wstart - 1] not in b" \n":
                wstart -= 1
            while wend < len(text) and text[wend] not in b" \n":
                wend += 1

            context = text[wstart - 1 : wend].decode("utf-8", errors="ignore")
            # Make sure context isn't just the token itself with a preceding space
            if context != f" {token_of_interest}":
                contexts_per_token[token_of_interest][context] += 1

    # Save results per token
    output_dir = Path("superstring_encodings")
    output_dir.mkdir(exist_ok=True)
    for token in tokens_of_interest:
        results = []
        for context, count in contexts_per_token[token].items():
            encoded_tokens = tokenizer.encode(context, add_special_tokens=False).tokens
            results.append(
                {
                    "token": token,
                    "context": context,
                    "context_count": count,
                    "encoded_tokens": encoded_tokens,
                }
            )

        df = pd.DataFrame(results)
        df.to_json(output_dir / f"{token}.jsonl", lines=True, orient="records") #encode filename to skip special characters

batch_size = 1000
for i in range(27000, len(vocab), batch_size):
    batch = vocab[i:i+batch_size]
    # Skip tokens already processed
    batch_to_process = [t for t in batch if not os.path.exists(f"superstring_encodings/{t}.jsonl")]
    if not batch_to_process:
        continue

    print(f"Processing batch {i//batch_size + 1}: {i + batch_size} of {len(vocab)} tokens.")
    search_tokens(batch_to_process)
