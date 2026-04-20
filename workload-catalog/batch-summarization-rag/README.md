# Batch Summarization RAG

This workload simulates Retrieval Augmented Generation (RAG) or batch summarization tasks, often involving large documents or vector search results.

## 1. Use Case and Distributions
**What it is**: Answering questions based on retrieved documents or summarizing large texts.
**Why the distributions**:
- **Input Sequence Length (ISL)**: Bimodal. Massive split between short vector QA (e.g., 2k tokens) and full document reads (e.g., 60k tokens).
- **Output Sequence Length (OSL)**: Normal. Usually follows strict formatting or JSON guidelines, leading to predictable lengths.
- **Number of Turns**: 1.

## 2. Reference Datasets
- **[LongBench-Pro](https://huggingface.co/datasets/caskcsg/LongBench-Pro)**: A benchmark for evaluating long-context capabilities, featuring tasks with long inputs, ideal for RAG and summarization testing.

## 3. System Impact
- **High Prefill Load**: Large inputs (especially in full doc reads) create a heavy load on the prefill phase.
- **Prefix Aware Routing**: Highly beneficial as retrieved documents might be shared across requests.
- **KV Cache Offloading**: Needed for the large contexts to avoid OOM.
