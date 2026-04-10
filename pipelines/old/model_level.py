import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ModelLevelPipeline:
    def __init__(self):
        model_name = "protectai/deberta-v3-base-prompt-injection"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def run(self, prompt):
        start = time.time()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()

        latency = time.time() - start

        # Label mapping (IMPORTANT)
        # 0 = safe, 1 = injection (unsafe)
        if pred == 1:
            result = "unsafe"
        else:
            result = "safe"

        return result, latency
    