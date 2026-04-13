# Deep Research Workload

This workload simulates a deep research agent that explores the web or documents to answer complex questions.

## 1. Use Case and Distributions
**What it is**: The system performs multi-step research, ingesting web pages or documents and generating logs and reports.
**Why the distributions**:
- **Input Sequence Length (ISL)**: Exponential. Context swells massively as loops accumulate more information.
- **Output Sequence Length (OSL)**: Bimodal. Generates many short traces/logs, followed by a massive final report.
- **Number of Turns**: Normal. Capped by system limits or token budgets.

## 2. Reference Datasets
- **[Mind2Web-Live](https://huggingface.co/datasets/iMeanAI/Mind2Web-Live)**: A dataset for web agents, containing real-world web interaction traces, useful for simulating the browsing and research behavior of deep research agents.

## 3. System Impact
- **Extreme KV Cache Pressure**: As context grows exponentially, memory management is critical.
- **KV Cache Offloading**: Essential to handle the massive context without running out of memory.
- **Varied Output Load**: Switching between short bursts and long generation requires dynamic load management.
