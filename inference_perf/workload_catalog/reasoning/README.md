# Reasoning Workload

This workload simulates complex reasoning tasks, such as solving math problems or logic puzzles, where the model generates a lot of internal "thinking" before producing an answer.

## 1. Use Case and Distributions
**What it is**: Solving hard problems requiring step-by-step reasoning.
**Why the distributions**:
- **Input Sequence Length (ISL)**: Log-Normal. Prompt sizes are generally small and precise.
- **Output Sequence Length (OSL)**: Exponential. Long right tail depending on complexity, as the model may generate thousands of thinking tokens.
- **Number of Turns**: 1 (Single turn).

## 2. Reference Datasets
- **[AIME 2025](https://huggingface.co/datasets/opencompass/AIME2025)**: A dataset containing challenging mathematics problems (American Invitational Mathematics Examination), requiring deep reasoning and step-by-step thinking.

## 3. System Impact
- **Compute Bound Decoding**: The system spends a lot of time generating tokens (thinking), making it compute-bound during decoding.
- **Low Prefill/Decode Ratio**: Small input, huge output.
- **Speculative Decoding Benefit**: High potential for speculative decoding to speed up generation.
