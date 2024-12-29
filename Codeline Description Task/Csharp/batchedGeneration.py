from transformers import AutoTokenizer, AutoModelForCausalLM
from methodtools import lru_cache
from tqdm import tqdm

import torch
import random
import numpy as np
import sys

sys.setrecursionlimit(50000)

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Class that provides methods to jumble AutoTokenizer's tokenizations
class TokenJumbler:
    def __init__(self, vocabulary, tokenizer):
        self.vocab = vocabulary
        self.tokenizer = tokenizer

    def generateDefaultTokenization(self, sentence):
        return self.tokenizer.tokenize(sentence)

    def jumble(self, tokens, target_length):
        res = []
        seeing = False
        for index, token in enumerate(tokens):
            """
            if (token == ":"):
                seeing = True
            if (token == "ĠA"):
                seeing = False
            if (seeing):
                res += list(token)
            else:
                res.append(token)"""
            res += list(token)
        #print(res)
        return res

# Interface class for handling model inference and token processing
class TransformersInterface:
    def __init__(self, model, vocabulary, tokenizer, device='cuda'):
        self.jumbler = TokenJumbler(vocabulary, tokenizer)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.vocab = vocabulary

        # Ensure the model is on the correct device
        self.model.to(self.device)
        self.model.eval()  # Set model to evaluation mode

    def generate_response(self, batch, max_new_tokens=2048, batch_size=2, target_length=1):
        length_ratios = []
        normal_batch_inputs = []
        perturbed_batch_inputs = []
        input_ids_prefix = self.tokenizer.encode(
            """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a programming assistant trained to analyze and interpret code snippets. When provided with a code snippet and a set of answer choices (A, B, C, or D), your task is to evaluate the code, determine its behavior, and select the answer that best describes this behavior. Your response must be a single letter: A, B, C, or D. Do not provide explanations or additional text unless explicitly requested.\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>""",
            add_special_tokens=False
        )
        input_ids_suffix = self.tokenizer.encode(
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            add_special_tokens=False
        )

        # Process each sentence in the batch
        for sentence in tqdm(batch, desc="Processing Sentences", unit="sentence"):
            if isinstance(sentence, list):
                sentence = sentence[0]
            normal_tokenized = self.tokenizer.tokenize(sentence)
            jumbled_tokenized = self.jumbler.jumble(normal_tokenized.copy(), target_length=target_length)
            print(normal_tokenized)
            print(jumbled_tokenized)
            #print("Original Tokens:", normal_tokenized)
            #print("Jumbled Segments:", jumbled_tokenized)
            #print(len(normal_tokenized))
            #print(len(jumbled_tokenized))
            #print("-------------------------------------")
            length_ratios.append(len(jumbled_tokenized) / len(normal_tokenized))
            for token in normal_tokenized:
                if token not in self.vocab:
                    print(f"Warning: Token '{token}' not found in vocabulary.")
            #print(len(jumbled_tokenized) / len(normal_tokenized))
            normal_input_ids = self.tokenizer.convert_tokens_to_ids(normal_tokenized)
            jumbled_input_ids = self.tokenizer.convert_tokens_to_ids(jumbled_tokenized)
            normal_sequence = input_ids_prefix + normal_input_ids + input_ids_suffix
            jumbled_sequence = input_ids_prefix + jumbled_input_ids + input_ids_suffix
            normal_batch_inputs.append(normal_sequence)
            perturbed_batch_inputs.append(jumbled_sequence)

        # Combine normal and perturbed inputs
        combined_input_ids = normal_batch_inputs + perturbed_batch_inputs

        # Determine the maximum sequence length in the combined batch
        max_length = max(len(seq) for seq in combined_input_ids)

        # Create padded input IDs and attention masks with left padding
        padded_input_ids = []
        attention_masks = []

        for seq in combined_input_ids:
            padding_length = max_length - len(seq)
            # Left pad the sequence
            padded_seq = [self.tokenizer.pad_token_id] * padding_length + seq
            attention_mask = [0] * padding_length + [1] * len(seq)
            padded_input_ids.append(padded_seq)
            attention_masks.append(attention_mask)

        # Convert to PyTorch tensors
        input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long).to(self.device)
        attention_mask_tensor = torch.tensor(attention_masks, dtype=torch.long).to(self.device)

        # Initialize lists to store responses
        responses = []

        # Process the batch in chunks based on batch_size
        index = 0
        for i in range(0, len(combined_input_ids), batch_size):
            batch_input_ids = input_ids_tensor[i:i + batch_size]
            batch_attention_mask = attention_mask_tensor[i:i + batch_size]
            index += 1
            print("Processing Batch:", index)

            # Generate responses with deterministic decoding (greedy)
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False  # Ensures deterministic decoding
                )

            # Decode the generated token IDs to text
            decoded_responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
            responses.extend(decoded_responses)

        # Split the responses back into normal and perturbed
        half = len(normal_batch_inputs)
        normal_responses = responses[:half]
        perturbed_responses = responses[half:]

        return normal_responses, perturbed_responses, length_ratios