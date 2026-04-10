import torch
import time
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class LlamaGuardPipeline:
    def __init__(self, shared_llm):
        self.llm = shared_llm
        self.model_id = "aditya02acharya/Llama-3.2-1B-Prompt-Injection-merged"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 定义模型文档中指定的 Token ID
        self.YES_TOKEN_ID = 9891   # 'yes'
        self.NO_TOKEN_ID  = 2201   # 'no'
        self.THRESHOLD    = 0.12   # 官方建议的最佳阈值

        print(f"Loading 1B Guardrail with threshold {self.THRESHOLD}...")
        
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.guard_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=quant_config,
            device_map="auto"
        )
        self.guard_model.eval()

    def run(self, prompt: str):
        start_time = time.time()

        # 构造官方建议的 System Prompt 格式
        messages = [
            {"role": "system", "content": "You are a prompt injection detector. Reply only yes or no."},
            {"role": "user", "content": prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(self.device)

        with torch.no_grad():
            output = self.guard_model.generate(
                **inputs, # 使用 ** 进行解包
                max_new_tokens=1, 
                do_sample=False,
                output_scores=True, 
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # --- 核心改进：提取 P(yes) 概率 ---
        logits = output.scores[0] # (batch, vocab)
        # 只取 no 和 yes 两个 token 的 logits 进行 softmax
        pair_logits = logits[:, [self.NO_TOKEN_ID, self.YES_TOKEN_ID]]
        probs = F.softmax(pair_logits.float(), dim=-1)
        p_yes = probs[0, 1].item() # 拿到 yes 的概率

        # 根据 0.12 阈值判定
        if p_yes >= self.THRESHOLD:
            latency = time.time() - start_time
            return f"[SECURITY BLOCKED] 1B Guard detected Injection (P={p_yes:.4f})", latency

        # 安全则调用 8B 生成模型
        output_text, _ = self.llm.run(prompt)
        total_latency = time.time() - start_time
        return output_text, total_latency