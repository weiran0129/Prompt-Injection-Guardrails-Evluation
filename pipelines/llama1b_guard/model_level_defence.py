import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class MergedModelPipeline:
    def __init__(self):
        # 这个是你找到的经过微调合并的模型
        self.model_id = "aditya02acharya/Llama-3.2-1B-Prompt-Injection-merged"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()

    def run(self, prompt):
        start_time = time.time()
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        # 1. 这里的 inputs 是一个包含 'input_ids' 和 'attention_mask' 的字典
        inputs = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt",
            return_dict=True # 明确要求返回字典格式
        ).to(self.model.device)

        with torch.no_grad():
            # 2. 使用 **inputs 进行解包，这样模型就能拿到正确的 Tensor 了
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=100, 
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 3. 解码时也要注意从 inputs['input_ids'] 的长度开始截取
        response_ids = outputs[0][inputs['input_ids'].shape[-1]:]
        output_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        latency = time.time() - start_time
        return output_text, latency