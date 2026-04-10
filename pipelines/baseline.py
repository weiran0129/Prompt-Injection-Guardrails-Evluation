from transformers import pipeline, BitsAndBytesConfig
import time
import torch
import os
import logging

class BaselinePipeline:
    def __init__(self, shared_pipe):
        use_cuda = torch.cuda.is_available()
        print(f"Using device: {'GPU' if use_cuda else 'CPU'}")

        self.llm = shared_pipe

    def run(self, prompt):
        start = time.time()
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.llm.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        with torch.no_grad():
            output_list = self.llm(
                formatted_prompt, 
                max_new_tokens=50,   
                do_sample=False,      
                truncation=True,     
                pad_token_id=self.llm.tokenizer.eos_token_id,
                return_full_text=False 
            )
            
        output = output_list[0]['generated_text'].strip()
        latency = time.time() - start
        
        return output, latency
    