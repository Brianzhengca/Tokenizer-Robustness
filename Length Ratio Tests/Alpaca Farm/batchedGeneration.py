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

# Generates random segments from a word, guaranteeing each segment exists within vocabulary
class generateSegments:
    def __init__(self, word, vocabulary, target_length=2):
        self.word = word
        self.vocab = vocabulary
        self.wordlen = len(word)
        self.target_length = target_length

    @lru_cache(maxsize=None)
    def countSegments(self, start):
        if start == self.wordlen:
            return 1
        total = 0
        for end in range(start + 1, self.wordlen + 1):
            if self.word[start:end] in self.vocab:
                total += self.countSegments(end)
        return total

    @lru_cache(maxsize=None)
    def minSegments(self, start):
        if start == self.wordlen:
            return 0
        min_seg = float('inf')
        for end in range(start + 1, self.wordlen + 1):
            if self.word[start:end] in self.vocab:
                seg = 1 + self.minSegments(end)
                if seg < min_seg:
                    min_seg = seg
        return min_seg if min_seg != float('inf') else 0

    @lru_cache(maxsize=None)
    def maxSegments(self, start):
        if start == self.wordlen:
            return 0
        max_seg = 0
        for end in range(start + 1, self.wordlen + 1):
            if self.word[start:end] in self.vocab:
                seg = 1 + self.maxSegments(end)
                if seg > max_seg:
                    max_seg = seg
        return max_seg

    def buildSegments(self, start, prevlength):
        if start == self.wordlen:
            return []

        remaining_segments = self.target_length - prevlength
        lower_bound = int(0.9 * remaining_segments)
        upper_bound = int(1.1 * remaining_segments)

        choices = []
        weights = []

        for end in range(start + 1, self.wordlen + 1):
            segment = self.word[start:end]
            if segment in self.vocab:
                min_seg = self.minSegments(end)
                max_seg = self.maxSegments(end)
                new_remaining = remaining_segments - 1
                if min_seg <= new_remaining <= max_seg:
                    choices.append(segment)
                    weights.append(max_seg - min_seg + 1)  # Prefer segments with more flexibility
                    #print(f"Valid segment '{segment}' from {start} to {end} added.")
                #else:
                    #print(f"Segment '{segment}' from {start} to {end} rejected. min_seg: {min_seg}, max_seg: {max_seg}, new_remaining: {new_remaining}")

        if not choices:
            #print(f"No valid segments found at position {start} with prevlength {prevlength}. Returning original token.")
            return [self.word[start:]]  # Fallback to the original token

        nextSegment = random.choices(choices, weights=weights, k=1)[0]
        #print(f"Selected segment '{nextSegment}' from position {start}.")
        return [nextSegment] + self.buildSegments(start + len(nextSegment), prevlength + 1)

    def generate(self):
        total_possible = self.countSegments(0)
        if total_possible == 0:
            #print(f"No possible segmentation for '{self.word}'.")
            return []  # No valid segmentation
        min_seg = self.minSegments(0)
        max_seg = self.maxSegments(0)
        if not (min_seg <= self.target_length <= max_seg):
            #print(f"Adjusting target_length from {self.target_length} to fit between {min_seg} and {max_seg}")
            self.target_length = min(max(self.target_length, min_seg), max_seg)
        return self.buildSegments(0, 0)

# Class that provides methods to jumble AutoTokenizer's tokenizations
class TokenJumbler:
    def __init__(self, vocabulary, tokenizer):
        self.vocab = vocabulary
        self.tokenizer = tokenizer

    def generateDefaultTokenization(self, sentence):
        return self.tokenizer.tokenize(sentence)

    def jumble(self, tokens, target_length):
        res = []
        for token in tokens:
            if (target_length == 6): # special target length reserved for character level tokenization
                res += list(token)
                continue
            segments = generateSegments(token, self.vocab, target_length=target_length).generate()
            if not segments:
                segments = [token]  # Fallback to original token if segmentation fails
            res.extend(segments)
            #res += list(token)
        '''
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
        '''
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
            """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>""",
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
            #print(normal_tokenized)
            #print(jumbled_tokenized)
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

def main():
    # Initialize the tokenizer and model
    model_name = 'meta-llama/Llama-3.1-8B-Instruct'  # Ensure this is the correct model name
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
        token="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"
    )

    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        token="your hf token"
    )

    # Extract vocabulary from tokenizer
    vocabulary = set(tokenizer.get_vocab().keys())

    # Initialize the interface
    interface = TransformersInterface(model, vocabulary, tokenizer)

    # Define your batch of sentences
    batch = [
        "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "Carlos is planting a lemon tree. The tree will cost $90 to plant. Each year it will grow 7 lemons, which he can sell for $1.5 each. It costs $3 a year to water and feed the tree. How many years will it take before he starts earning money on the lemon tree?",
        "Kyle bought last year's best-selling book for $19.50. This is with a 25 percent discount from the original price. What was the original price of the book?",
        "Marissa is hiking a 12-mile trail. She took 1 hour to walk the first 4 miles, then another hour to walk the next two miles. If she wants her average speed to be 4 miles per hour, what speed (in miles per hour) does she need to walk the remaining distance?",
        "A candle melts by 2 centimeters every hour that it burns. How many centimeters shorter will a candle be after burning from 1:00 PM to 5:00 PM?"
    ]

    # Generate responses
    normal_responses, perturbed_responses, length_ratios = interface.generate_response(batch, max_new_tokens=512, batch_size=2)
    print(length_ratios)

    # Optional: Print responses
    for i, (normal, perturbed) in enumerate(zip(normal_responses, perturbed_responses)):
        print(f"Sentence {i + 1} Normal Response:\n{normal}\n")
        print(f"Sentence {i + 1} Perturbed Response:\n{perturbed}\n")
        print("=" * 50)

if __name__ == "__main__":
    main()
