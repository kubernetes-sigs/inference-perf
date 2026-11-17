# Code Generation Workload

This workload simulates code generation tasks, typically within an IDE or as part of a development workflow.

## 1. Use Case and Distributions
**What it is**: Assisting developers by generating code snippets, full files, or explaining code.
**Why the distributions**:
- **Input Sequence Length (ISL)**: Uniformly High. Starts massive due to repository maps, project context, and stays massive.
- **Output Sequence Length (OSL)**: Log-Normal. Lots of small function edits, with an occasional full-file rewrite creating a long tail.
- **Number of Turns**: Normal. Reflects iterative test-fail-fix cycles.

## 2. Reference Datasets
- **[CoderForge Preview](https://huggingface.co/datasets/togethercomputer/CoderForge-Preview)**: A dataset for code generation tasks, likely containing code snippets and prompts.
- **[Qwen Bailian Usage Traces](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon)**: Anonymous usage traces that may include code generation requests, useful for understanding real-world distributions.


## 3. System Impact
- **High Memory Pressure**: Massive input contexts require significant memory for the KV cache.
- **Prefix Caching Benefit**: Highly repetitive system prompts and context make this workload ideal for prefix caching.
- **Compute Bound Prefill**: Large inputs mean the prefill phase is compute-heavy.
