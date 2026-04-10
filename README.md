# Prompt Injection Guardrails Evaluation

Evaluating System-Level Guardrails Against Prompt Injection: Architecture, Robustness, and Latency Trade-offs

## Overview

This project evaluates multiple prompt injection defense mechanisms for Large Language Model (LLM) systems. The goal is to systematically analyze the trade-offs between:

- Security (Attack Success Rate, ASR)
- Utility (False Refusal Rate, FRR)
- Efficiency (Latency)

We benchmark four system-level guardrail architectures integrated with a strong baseline model (Llama-3-8B-Instruct).

## Guardrail Architectures

The following guardrails are evaluated:

- Baseline: Llama-3-8B-Instruct (no guardrail)
- Regex: Rule-based deterministic filtering
- DeBERTa: Transformer-based binary classifier
- Llama-3.2-1B: Local LLM-as-judge (discriminative)
- GPT-4o-mini: API-based LLM-as-judge

## Dataset

We construct a benchmark dataset of 1,000 samples from two public sources:

- neuralchemy/Prompt-injection-dataset
- JasperLS/prompt-injections

### Distribution

- 550 direct prompt injection samples
- 30 indirect prompt injection samples
- 220 additional adversarial samples
- 200 benign (safe) prompts

This results in an attack-heavy dataset with 80% unsafe and 20% safe prompts, designed to stress-test guardrail robustness.

### Preprocessing

- Maximum input length: 500 characters
- Dataset cleaned, merged, and shuffled
- Inputs normalized to reduce noise

## Experimental Procedure

Each prompt is processed through a standardized pipeline:

1. The guardrail (if enabled) analyzes the input prompt
2. If classified as unsafe:
   - The pipeline returns `[SECURITY BLOCKED]`
3. If classified as safe:
   - The prompt is passed to the Llama-3-8B model
   - A response is generated
4. The output is evaluated using an LLM-as-judge (GPT-4o-mini)

Latency is measured as the total time for guardrail + generation.

## Evaluation Metrics

We evaluate performance using the following metrics:

- Attack Success Rate (ASR): Percentage of unsafe prompts that bypass the guardrail
- False Refusal Rate (FRR): Percentage of safe prompts incorrectly blocked
- Latency: End-to-end execution time (seconds)
- Accuracy, Precision, Recall, F1 score

## Results

| Pipeline           | Accuracy | Precision | Recall | F1    | ASR    | FRR   | Latency (s) |
|------------------|----------|-----------|--------|-------|--------|-------|-------------|
| Baseline          | 0.336    | 0.993     | 0.171  | 0.292 | 82.9%  | 0.5%  | 3.813       |
| Regex             | 0.342    | 0.993     | 0.179  | 0.303 | 82.1%  | 0.5%  | 3.705       |
| DeBERTa           | 0.885    | 0.994     | 0.861  | 0.923 | 13.9%  | 2.0%  | 1.676       |
| Llama-1B          | 0.873    | 0.880     | 0.974  | 0.925 | 2.6%   | 53.0% | 0.627       |
| GPT-4o-mini       | 0.899    | 0.993     | 0.880  | 0.933 | 12.0%  | 2.5%  | 1.990       |

## Key Findings

1. Regex-based guardrails provide minimal improvement over baseline and are insufficient against sophisticated attacks.

2. DeBERTa and GPT-based guardrails offer the best balance between robustness and usability, achieving low ASR with minimal impact on FRR.

3. Llama-1B (LLM-based guardrail) achieves the lowest ASR but suffers from extremely high FRR, making it impractical in real-world applications.

4. Strong alignment in model-level defenses can lead to over-refusal, referred to as the "Alignment & Defense Tax".

## Project Structure
