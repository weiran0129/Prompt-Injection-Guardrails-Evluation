import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, BitsAndBytesConfig

class DebertaGuardrailPipeline:
    def __init__(self, shared_pipe):
        model_name = "protectai/deberta-v3-base-prompt-injection"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.classifier = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.classifier.eval()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier.to(self.device)

        #baseline: Llama-3-8B
        use_cuda = torch.cuda.is_available()
        self.llm = shared_pipe

    def run(self, prompt):
        start_time = time.time()

        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.classifier(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()

        # 0 = safe, 1 = unsafe
        if pred == 1:
            # unsafe, dont go inside llama 3b
            latency = time.time() - start_time
            return "[SECURITY BLOCKED] Guardrail detection: Unsafe Prompt", latency

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
            
        response = output_list[0]['generated_text'].strip()
        total_latency = time.time() - start_time
        
        return response, total_latency