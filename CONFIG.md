# Inference Performance Benchmark Configuration Guide

## Table of Contents

1. [Overview](#overview)
2. [Configuration Structure](#configuration-structure)
   - [API Configuration](#api-configuration)
   - [Data Generation](#data-generation)
   - [Load Configuration](#load-configuration)
   - [Model Server](#model-server)
   - [Metrics Collection](#metrics-collection)
   - [Reporting](#reporting)
   - [Storage](#storage)
   - [Tokenizer](#tokenizer)
3. [Full Configuration Examples](#full-configuration-examples)

## Overview

This document provides complete documentation for all configuration options available in the Kubernetes Inference Performance Benchmark tool.

## Configuration Structure

### API Configuration

Controls the API interaction behavior:

```yaml
api:
  type: completion  # API type (completion|chat)
  streaming: false  # Enable/disable streaming
```  

### Data Generation

Configures the test data generation methodology:

```yaml
data:
  type: mock|shareGPT|synthetic|random|shared_prefix  # Data generation type
  input_distribution:                                 # For synthetic/random types
    min: 10                                           # Minimum prompt length (tokens)
    max: 100                                          # Maximum prompt length
    mean: 50                                          # Average length
    std: 10                                           # Standard deviation
    total_count: 100                                  # Total prompts to generate
  output_distribution:                                # Same structure as input_distribution
    min: 10
    max: 100
    mean: 50
    std: 10
    total_count: 100
  shared_prefix:              # For shared_prefix type
    num_groups: 10            # Number of shared prefix groups
    num_prompts_per_group: 10 # Unique questions per group
    system_prompt_len: 100    # Shared prefix length (tokens)
    question_len: 50          # Question length (tokens)
    output_len: 50            # Target output length (tokens)  
```

### Load Configuration

Defines the benchmarking load pattern:

```yaml
load:
  type: constant|poisson            # Load pattern type
  interval: 1.0                     # Seconds between request batches
  stages:                           # Load progression stages
    - rate: 1                       # Requests per second
      duration: 30                  # Seconds to maintain this rate
  num_workers: 4                    # Concurrent worker threads (default: CPU_cores/2)
  worker_max_concurrency: 10        # Max concurrent requests per worker
  worker_max_tcp_connections: 2500  # Max TCP connections per worker
```

### Model Server

Configures connection to the model serving backend:

```yaml
server:
  type: vllm                                          # Currently only vLLM supported
  model_name: "HuggingFaceTB/SmolLM2-135M-Instruct"   # Required model identifier
  base_url: "http://0.0.0.0:8000"                     # Required server endpoint
  ignore_eos: true                                    # Whether to ignore End-of-Sequence tokens
  api_key: ""                                         # Optional API key for authenticated endpoints
```

### Metrics Collection

Sets up performance metrics collection:

```yaml
metrics:
  type: prometheus|default        # Metrics backend type
  prometheus:                     # Required when type=prometheus
    url: "http://localhost:9090"  # Prometheus server URL
    scrape_interval: 15           # Metrics scrape interval (seconds)
    google_managed: false         # Whether using Google Managed Prometheus
    filters: []                   # List of metric names to collect
```

### Reporting

Controls benchmark report generation:

```yaml
report:
  request_lifecycle:
    summary: true       # Generate high-level summary
    per_stage: true     # Include breakdown by load stage
    per_request: false  # Enable detailed per-request logs (verbose)
  prometheus:
    summary: true       # Include Prometheus metrics summary
    per_stage: false    # Disable Prometheus stage breakdown
```

### Storage

Configures storage for benchmark results:

```yaml
storage:
  local_storage:
    path: "reports-{timestamp}"       # Local directory path
    report_file_prefix: null          # Optional filename prefix
  google_cloud_storage:               # Optional GCS configuration
    bucket_name: "your-bucket-name"   # Required GCS bucket
    path: "reports-{timestamp}"       # Optional path prefix
    report_file_prefix: null          # Optional filename prefix
  simple_storage_service:
    bucket_name: "your-bucket-name"   # Required S3 bucket
    path: "reports-{timestamp}"       # Optional path prefix
    report_file_prefix: null          # Optional filename prefix
```

### Tokenizer

Optional tokenizer configuration for specialized tokenization:

```yaml
tokenizer:
  pretrained_model_name_or_path: "model-id"   # Required model path
  trust_remote_code: true                     # Whether to trust custom tokenizer code
  token: ""                                   # HuggingFace access token for private models
```

## Full Configuration Examples

### Minimal Configuration

```yaml
data:
  type: shareGPT
load:
  type: constant
  stages:
  - rate: 1
    duration: 30
api: 
  type: chat
server:
  type: vllm
  model_name: HuggingFaceTB/SmolLM2-135M-Instruct
  base_url: http://0.0.0.0:8000
```

### Advanced Configuration

```yaml
load:
  type: constant
  stages:
  - rate: 1
    duration: 30
api: 
  type: completion
server:
  type: vllm
  model_name: HuggingFaceTB/SmolLM2-135M-Instruct
  base_url: http://0.0.0.0:8000
  ignore_eos: true
tokenizer:
  pretrained_model_name_or_path: HuggingFaceTB/SmolLM2-135M-Instruct
data:
  type: random
  input_distribution:
    min: 10             # min length of the synthetic prompts
    max: 100            # max length of the synthetic prompts
    mean: 50            # mean length of the synthetic prompts
    std: 10             # standard deviation of the length of the synthetic prompts
    total_count: 100    # total number of prompts to generate to fit the above mentioned distribution constraints
  output_distribution:
    min: 10             # min length of the output to be generated
    max: 100            # max length of the output to be generated
    mean: 50            # mean length of the output to be generated
    std: 10             # standard deviation of the length of the output to be generated
    total_count: 100    # total number of output lengths to generate to fit the above mentioned distribution constraints
metrics:
  type: prometheus
  prometheus:
    url: http://localhost:9090
    scrape_interval: 15
report:
  request_lifecycle:
    summary: true
    per_stage: true
    per_request: true
```
