import torch
import pandas as pd
import os
from transformers.utils import logging
from transformers import pipeline, BitsAndBytesConfig
   
os.environ["HF_TOKEN"] = ""
os.environ["OPENAI_API_KEY"] = ""

from data.dataset_loader import load_dataset
from pipelines.baseline import BaselinePipeline
from pipelines.deberta_guardrail import DebertaGuardrailPipeline
from pipelines.regex_guardrail import RegexGuardPipeline
from pipelines.llama1b_guardrail import LlamaGuardPipeline
from pipelines.gpt_guardrail import GPTGuardrailPipeline
#from pipelines.model_level_defence import MergedModelPipeline
from evaluation.metrics import evaluate_baseline, evaluate_system

logging.set_verbosity_error()


def run_pipeline(pipeline, dataset, name, output_csv, evaluator):
    results = []
    print(f"Starting pipeline: {name}")

    for i, item in enumerate(dataset):
        prompt = item["prompt"]
        label = item["label"]

        output, latency = pipeline.run(prompt)

        print(f"[{name}] {i+1}/{len(dataset)} | {latency:.2f}s")

        results.append({
            "id": i,
            "prompt": prompt,
            "label": label,
            "output": output,
            "latency": latency,
        })

    df = pd.DataFrame(results)

    # 3. Compute metrics based on which pipeline
    metrics = evaluator(df)

    # Add metrics to every row
    for k, v in metrics.items():
        df[k] = v

    df["pipeline"] = name

    df.to_csv(output_csv, index=False)

    print(f"\nSaved to {output_csv}")
    print("Metrics:", metrics)
    return df

if __name__ == "__main__":
    dataset_path = "src/test_data.csv"
    dataset = load_dataset(dataset_path)
    '''
    base_model_id = "meta-llama/Llama-3.2-1B-Instruct" 
    
    print(f"Loading Base Model: {base_model_id}")
    shared_pipe = pipeline(
        "text-generation",
        model=base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    '''
    print("Loading shared LLM instance...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    shared_pipe = pipeline(
        "text-generation",
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        device_map="auto",
        model_kwargs={"quantization_config": quant_config}
    )

    # --- 1. Baseline ---
    print("\n>>> Running: Baseline")
    baseline = BaselinePipeline(shared_pipe)
    run_pipeline(baseline, dataset, "baseline", "./results/baseline_results.csv", evaluate_baseline)

    #print("\n>>> Running: Llama-1B Guardrail")
    #merged_model = MergedModelPipeline()
    #run_pipeline(merged_model, dataset, "model-level-1b", "./results/model-level-1b_results.csv", evaluate_baseline)

    # --- 2. GPT Guardrail ---
    print("\n>>> Running: GPT Guardrail")
    gpt_guard = GPTGuardrailPipeline(baseline)
    run_pipeline(gpt_guard, dataset, "gpt_api_based", "./results/gpt_guard_results.csv", evaluate_baseline)
    del gpt_guard 
    torch.cuda.empty_cache()

    # --- 3. Llama-1B Guardrail ---
    print("\n>>> Running: Llama-1B Guardrail")
    llama_1b_guard = LlamaGuardPipeline(baseline) 
    run_pipeline(llama_1b_guard, dataset, "llama_1b_llm_based", "./results/llama_1b_results.csv", evaluate_baseline)
    del llama_1b_guard 
    torch.cuda.empty_cache()

    # --- 4. DeBERTa Guardrail ---
    print("\n>>> Running: DeBERTa Guardrail")
    deberta_guardrail = DebertaGuardrailPipeline(shared_pipe)
    run_pipeline(deberta_guardrail, dataset, "deberta_guardrail", "./results/deberta_results.csv", evaluate_system)
    del deberta_guardrail
    torch.cuda.empty_cache()

    # --- 5. Regex Guardrail ---
    print("\n>>> Running: Regex Guardrail")
    regex_guardrail = RegexGuardPipeline(shared_pipe)
    run_pipeline(regex_guardrail, dataset, "regex_guardrail", "./results/regex_results.csv", evaluate_system)

    print("\n[ALL EXPERIMENTS COMPLETED]")

'''
if __name__ == "__main__":
    dataset_path = "src/test_data.csv"
    dataset = load_dataset(dataset_path)

    print("Loading shared LLM instance...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    shared_pipe = pipeline(
        "text-generation",
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        device_map="auto",
        model_kwargs={"quantization_config": quant_config}
    )

    baseline = BaselinePipeline(shared_pipe)

    configs = [
        {
            "name": "baseline",
            "obj": baseline,
            "path": "./results/baseline_results.csv",
            "eval": evaluate_baseline
        },
        {
            "name": "regex_guardrail",
            "obj": RegexGuardPipeline(shared_pipe),
            "path": "./results/regex_results.csv",
            "eval": evaluate_system
        },
        {
            "name": "gpt_api_based",
            "obj": GPTGuardrailPipeline(baseline),
            "path": "./results/gpt_guard_results.csv",
            "eval": evaluate_baseline
        },
        {
            "name": "deberta_guardrail",
            "obj": DebertaGuardrailPipeline(shared_pipe),
            "path": "./results/deberta_results.csv",
            "eval": evaluate_system
        },
        {
            "name": "llama_1b_llm_based",
            "obj": LlamaGuardPipeline(baseline), 
            "path": "./results/llama_1b_results.csv",
            "eval": evaluate_baseline
        }
    ]

    for cfg in configs:
        try:
            print(f"\n>>> Preparing to run: {cfg['name']}")
            run_pipeline(cfg["obj"], dataset, cfg["name"], cfg["path"], cfg["eval"])
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error running {cfg['name']}: {e}")
    print("\n[ALL EXPERIMENTS COMPLETED]")


   
    baseline = BaselinePipeline(shared_pipe)
    run_pipeline(baseline, dataset, "baseline", "./results/baseline_results.csv", evaluate_baseline)

    gpt_guard = GPTGuardrailPipeline(baseline)
    run_pipeline(gpt_guard, dataset, "gpt_api_based", "./results/gpt_guard_results.csv", evaluate_baseline)

    llama_1b_guard = LlamaGuardPipeline(baseline) 
    run_pipeline(llama_1b_guard, dataset, "llama_1b_llm_based", "./results/llama_1b_results.csv", evaluate_baseline)

    deberta_guardrail = DebertaGuardrailPipeline(shared_pipe)
    run_pipeline(deberta_guardrail, dataset, "deberta_guardrail", "./results/deberta_results.csv", evaluate_system)

    regex_guardrail = RegexGuardPipeline(shared_pipe)
    run_pipeline(regex_guardrail, dataset, "regex_guardrail", "./results/regex_results.csv", evaluate_system)
    '''
    