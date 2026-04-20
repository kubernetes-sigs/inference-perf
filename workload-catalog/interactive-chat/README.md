# Interactive Chat Workload

This workload simulates a multi-turn chat conversation between a user and an AI assistant.

## 1. Use Case and Distributions
**What it is**: Standard conversational AI interaction, where users ask questions and get answers in a multi-turn session.
**Why the distributions**:
- **Input Sequence Length (ISL)**: Log-Normal. Most sessions start with short prompts, but a long tail exists for power users or copy-pasted context.
- **Output Sequence Length (OSL)**: Normal. Standard conversational lengths are relatively consistent.
- **Number of Turns**: Log-Normal. Most users ask 1-3 questions, some maintain long threads.

## 2. Reference Datasets
- **[Qwen Bailian Usage Traces](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon)**: Anonymous real-world usage traces from the Qwen Bailian platform, reflecting actual user chat behavior and distributions.


## 3. System Impact
- **High Concurrency**: Requires handling many simultaneous users with low latency expectations.
- **KV Cache Growth**: KV cache grows with each turn, putting pressure on memory over time if not managed (e.g., via prefix caching).
- **Latency Sensitivity**: Time-to-first-token (TTFT) and inter-token latency (ITL) are critical for user experience.
