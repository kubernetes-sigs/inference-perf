# Workload Catalog

This directory contains a catalog of real-world benchmarking workloads covering a wide range of use cases that are currently relevant in the AI inference space.

## Goal

The aim of this catalog is to standardize and provide a way to reproducibly benchmark real-world workloads. Regardless of the benchmark harness used underneath, these workload catalog entries should contain enough detail to allow for reproduction of the performance characteristics.

## Structure

Each workload in the catalog is organized into a directory containing:

1.  **`config.json`**: Describes the use case related parameters in detail (e.g., input/output sequence lengths, distributions, number of turns). This file is intended to be harness-agnostic and purely descriptive of the workload's statistical profile.
2.  **`inference-perf.yaml`**: A benchmark configuration file specific to `inference-perf` that can be used to run the workload directly.
3.  **`README.md`**: Documentation explaining the use case, rationale for distributions, reference datasets, and system impact.

## Standardization

When these workloads are run through a harness (like `inference-perf`), they should produce a report in the upcoming standard benchmarking format. This will make it easier to compare results across different serving stacks and hardware configurations.

## Available Workloads

- **interactive-chat**: Simulates a multi-turn chat conversation.
- **code-generation**: Simulates heavy code generation tasks.
- **deep-research**: Simulates a long-running research agent with massive context.
- **reasoning**: Simulates complex step-by-step reasoning tasks.
- **batch-summarization-rag**: Simulates RAG or batch summarization with long inputs.
- **batch-synthetic-data-generation**: Simulates high-throughput synthetic data generation.
