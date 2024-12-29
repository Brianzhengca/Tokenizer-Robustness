from transformers import AutoTokenizer, AutoModelForCausalLM
from tokenizers import Tokenizer
from methodtools import lru_cache
from tqdm import tqdm

import torch
import pickle as pkl
import random
import numpy as np

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Generates random segments from a word, guaranteeing each segment exists within vocabulary
class generateSegments:

    # Pre: string word representing the word to be segmented, set vocabulary representing 
    #       the vocabulary to check the segments against
    # Post: creates a new generateSegments Object, does not output.
    def __init__(self, word, vocabulary):
        self.word = word
        self.vocab = vocabulary
        self.wordlen = len(word)
    
    # Pre: accepts an integer start representing the index to start from
    # Post: returns the number of segments that potentially start from start index in self.word.
    #       Cached method. Uses methodtools.lru_cache instead of functools.cache to avoid 
    #       memory leaks
    @lru_cache()
    def countSegments(self, start):
        if start == self.wordlen:
            return 1
        total = 0
        for end in range(start + 1, self.wordlen + 1):
            if self.word[start:end] in self.vocab:
                total += self.countSegments(end)
        return total
    
    # Pre: accepts an integer start representing the starting index that we want to build 
    #       a segment from 
    # Post: builds the segmentation for self.word. At each recursive step, randomly choose 
    #       a child to visit, weighed by the size of the subtree rooted at the ith child
    def buildSegments(self, start):
        if start == self.wordlen:
            return []
        choices = []
        weights = []
        for end in range(start + 1, self.wordlen + 1):
            segment = self.word[start:end]
            if segment in self.vocab:
                count = self.countSegments(end)
                if count > 0:
                    choices.append(segment)
                    weights.append(count)
        if not choices:
            return []
        nextSegment = random.choices(choices, weights=weights, k=1)[0]
        return [nextSegment] + self.buildSegments(start + len(nextSegment))

    # Post: generate a random segmentation where each segment exists in self.vocab.
    #       Returns the segmentation
    def generate(self):
        total = self.countSegments(0)
        if total == 0:
            return [] # If there is no valid way to segment self.word 
        return self.buildSegments(0)

# class that provides methods that support jumbling AutoTokenizer's tokenizations
class TokenJumbler:

    # Pre: accepts a set vocabulary that we would check segments against and an AutoTokenizer 
    #       object to create tokenzations for a certain string 
    # Post: initializes a TokenJumbler object, Does not output.
    def __init__(self, vocabulary, tokenizer):
        self.vocab = vocabulary
        self.tokenizer = tokenizer
    
    # Pre: accepts a string sentence representing the sentence to be tokenized
    # Post: generates and returns a list consisting of word tokens based on self.tokenizer
    def generateDefaultTokenization(self, sentence):
        return self.tokenizer.tokenize(sentence)
    
    # Pre: accepts a list tokens representing the tokens to be jumbled 
    # Post: jumbles the tokens randomly according to generateSegments() and return a list
    #       consisting of the jumbled tokens
    def jumble(self, tokens):
        #lastToken = tokens[-1]
        #tokens.pop()
        #tokens += list(lastToken)
        #return tokens
        res = []
        for token in tokens:
            res += generateSegments(token, self.vocab).generate()
        return res

# Interface class for handling model inference and token processing
class TransformersInterface:
    def __init__(self, model, vocabulary, tokenizer, device='cuda'):
        self.jumbler = TokenJumbler(vocabulary, tokenizer)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Ensure the model is on the correct device
        self.model.to(self.device)
        self.model.eval()  # Set model to evaluation mode
    
    def generate_response(self, batch, max_new_tokens=512, batch_size=2):
        normal_batch_inputs = []
        perturbed_batch_inputs = []
        input_ids_prefix = self.tokenizer.encode("<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>", add_special_tokens=False)
        input_ids_suffix = self.tokenizer.encode("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)
        #input_ids_prefix = self.tokenizer.encode("<bos>\n\n", add_special_tokens=False) # Gemma
        #input_ids_suffix = self.tokenizer.encode("<end_of_turn><start_of_turn>model\n\n", add_special_tokens=False)
        
        # Process each sentence in the batch
        for sentence in tqdm(batch, desc="Processing Sentences", unit="sentence"):
            if isinstance(sentence, list):
                sentence = sentence[0]
            normal_tokenized = self.tokenizer.tokenize(sentence)
            jumble_tokenized = self.jumbler.jumble(normal_tokenized.copy())
            normal_input_ids = self.tokenizer.convert_tokens_to_ids(normal_tokenized)
            jumbled_input_ids = self.tokenizer.convert_tokens_to_ids(jumble_tokenized)
            normal_sequence = input_ids_prefix + normal_input_ids + input_ids_suffix
            jumbled_sequence = input_ids_prefix + jumbled_input_ids + input_ids_suffix
            #normal_sequence = normal_input_ids + input_ids_suffix
            #jumbled_sequence = jumbled_input_ids + input_ids_suffix #Gemma only
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
            batch_input_ids = input_ids_tensor[i:i+batch_size]
            batch_attention_mask = attention_mask_tensor[i:i+batch_size]
            index += 1
            print("Processing Batch:", index)
            
            # Generate responses with deterministic decoding (greedy)
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False    # Ensures deterministic decoding
                )
            
            # Decode the generated token IDs to text
            decoded_responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
            responses.extend(decoded_responses)
        
        # Split the responses back into normal and perturbed
        half = len(normal_batch_inputs)
        normal_responses = responses[:half]
        perturbed_responses = responses[half:]
        
        return normal_responses, perturbed_responses