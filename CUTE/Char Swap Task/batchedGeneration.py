from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np

import random

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

class Generator:
    def __init__(self, data, tokenizer, model, batch_size=5):
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.model = model
        self.model.to('cuda')
        self.data = self.apply_chat_format(data)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
        self.batched_data = list(self.pad())
    def pad(self):
        dataloader = DataLoader(self.data, batch_size=self.batch_size, shuffle=False)
        for batch in dataloader:
            yield self.tokenizer(batch, padding=True, return_tensors="pt")
    def apply_chat_format(self, raw_data):
        messages = []
        for datapoint in raw_data:
            messages.append(self.tokenizer.apply_chat_template([{"role":"user", "content":datapoint}], tokenize=False, add_generation_prompt=True))
        return messages
    def generate(self):
        batch_outputs = []
        for batch in tqdm(self.batched_data):
            output = self.model.generate(
                batch['input_ids'].cuda(),
                attention_mask=batch["attention_mask"].cuda(),
                max_new_tokens=200,
                do_sample=False,
                top_p=1,
                temperature=0)
            batch_outputs.extend(self.tokenizer.batch_decode(output))
        return batch_outputs
        
    
def main():
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data = [
        "Hello, how are you today?",
        "I am doing fine, thank you, and you?",
        "Oh wow! That is very good to hear"
    ]
    generator = Generator(data, tokenizer, model, batch_size=2)

if __name__ == "__main__":
    main()