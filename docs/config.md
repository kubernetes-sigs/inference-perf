# Configuration Guide

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
4. [Advanced Use Cases](#advanced-use-cases)
   - [OpenTelemetry Trace Replay](#opentelemetry-trace-replay)

## Overview

This document provides complete documentation for all configuration options available in the Kubernetes Inference Performance Benchmark tool.

## Configuration Structure

### API Configuration

Controls the API interaction behavior. If SLO headers are present, each request is evaluated for SLO compliance and SLO-related metrics are reported:

```yaml
api:
  type: completion             # API type (completion|chat). completion is default since chat may require extra server config
  streaming: true             # Enable streaming for TTFT, ITL, and TPOT metrics
  headers:                     # Optional custom HTTP headers
    x-inference-model: llama
    x-routing-strategy: round-robin
    x-slo-tpot-ms: "2"
    x-slo-ttft-ms: "1000"
  slo_unit: "ms"               # Optional SLO unit (e.g., ms, s), default is ms
  slo_tpot_header: "x-slo-tpot-ms"        # Optional header name for TPOT SLO Header, default is x-slo-tpot-ms
  slo_ttft_header: "x-slo-ttft-ms"        # Optional header name for TTFT SLO Header, default is x-slo-ttft-ms
```  

### Data Generation

Configures the test data generation methodology:

```yaml
data:
  type: mock|shareGPT|synthetic|random|shared_prefix|cnn_dailymail|billsum_conversations|infinity_instruct|otel_trace_replay # Data generation type
  path: ./data/shareGPT/ShareGPT_V3_unfiltered_cleaned_split.json # For shareGPT type, path where dataset to be used is present. Path needs to be set for cnn_dailymail, billsum_conversations and infinity_instruct as well
  input_distribution:                                 # For synthetic/random types
    min: 10                                           # Minimum prompt length (tokens)
    max: 100                                          # Maximum prompt length
    mean: 50                                          # Average length
    std_dev: 10                                       # Standard deviation
    total_count: 100                                  # Total prompts to generate
  output_distribution:                                # Same structure as input_distribution
    min: 10
    max: 100
    mean: 50
    std_dev: 10
    total_count: 100
  shared_prefix:              # For shared_prefix type
    num_groups: 10            # Number of shared prefix groups
    num_prompts_per_group: 10 # Unique questions per group
    system_prompt_len: 100    # Shared prefix length (tokens)
    question_len: 50          # Default question length (tokens), used when question_distribution is absent
    output_len: 50            # Default output length (tokens), used when output_distribution is absent
    question_distribution:    # Optional: distribution for question lengths (overrides question_len)
      min: 10
      max: 1024
      mean: 50
      std_dev: 5
    output_distribution:      # Optional: distribution for output lengths (overrides output_len)
      min: 10
      max: 1024
      mean: 50
      std_dev: 5
```

**Note:** For `otel_trace_replay` type, see the [OpenTelemetry Trace Replay](#opentelemetry-trace-replay) section for complete configuration details.

### Load Configuration

Defines the benchmarking load pattern:

```yaml
load:
  type: constant|poisson|concurrent|trace_session_replay # Load pattern type
  interval: 1.0                     # Seconds between request batches
  stages:                           # Load progression stages
    - rate: 1                       # Requests per second (CONSTANT or POISSON LOADS)
      duration: 30                  # Seconds to maintain this rate (CONSTANT or POISSON LOADS)
      concurrency_level: 3          # Level of concurrency/number of worker threads (CONCURRENT LOADS)
      num_requests: 40              # Number of requests to be processed by concurrency_level worker threads (CONCURRENT LOADS)
  num_workers: 4                    # Concurrent worker threads (default: CPU_cores)
  worker_max_concurrency: 10        # Max concurrent requests per worker
  worker_max_tcp_connections: 2500  # Max TCP connections per worker
  base_seed: 12345                  # Optional: base random seed for reproducibility (default: current time in ms)
  lora_traffic_split:               # Optional: MultiLoRA traffic splitting
    - name: adapter_1               # LoRA adapter name
      split: 0.5                    # Traffic weight (must sum to 1.0)
    - name: adapter_2
      split: 0.5
```

**Note:** `trace_session_replay` load type has different stage parameters. See [OpenTelemetry Trace Replay](#opentelemetry-trace-replay) for configuration details.

#### Load Sweeps

Defines the preprocessing phase to determine load based on
target service saturation.

```yaml
load:
  type: constant|poisson
  interval: 15
  sweep:                        # Automatically determine saturation point of the target service and generate stages
    type: linear|geometric      # Produce a linear distribution [1.0, saturation] of rates for num_stages or geometric distribution clustered around the saturation point
    timeout: 60                 # Length of time to run load to determine saturation
    num_stages: 5               # Number of stages to generate
    stage_duration: 180         # Duration of each generated stage
    saturation_percentile: 95   # Percentile of sampled rates to select as saturation point
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
    google_managed: false         # Whether using Google Managed Prometheus (see 'Google Managed Prometheus (GMP) Requirements' section)
    filters: []                   # List of metric names to collect
```

#### Google Managed Prometheus (GMP) Requirements

When setting `google_managed: true`, `inference-perf` queries the GMP API directly. You must configure [Application Default Credentials (ADC)](https://cloud.google.com/docs/authentication/application-default-credentials) in your environment with sufficient permissions.

1. **Required Permissions**
   The identity used by ADC must have the Monitoring Viewer role:
   * `roles/monitoring.viewer`

2. **Environment Configuration**
   * **GKE Cluster:** Ensure the Pod is running with [Workload Identity](https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity) enabled and linked to a Google Service Account (GSA) with the required role.
   * **GCE VM:** Ensure the [VM's attached Service Account](https://cloud.google.com/compute/docs/access/service-accounts#associating_a_service_account_to_an_instance) has the required role.
   * **Local Development:** Authenticate using your user credentials:
     ```bash
     gcloud auth application-default login
     ```
     > **Note:** Your personal user account must have the `monitoring.viewer` role on the target GCP project.

**Common Error:**
Failing to configure these permissions will result in API errors similar to:
```text
ERROR - error executing query: 403 Client Error: Forbidden for url: [https://monitoring.googleapis.com/v1/projects/](https://monitoring.googleapis.com/v1/projects/)...
```

### Reporting

Controls benchmark report generation:

```yaml
report:
  request_lifecycle:
    summary: true             # Generate high-level summary
    per_stage: true           # Include breakdown by load stage
    per_request: false        # Enable detailed per-request logs (verbose)
    per_adapter: false        # Generate metrics grouped by LoRA adapter
    per_adapter_stage: false  # Generate metrics grouped by adapter and stage
    percentiles: [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9] # List of percentiles to calculate
  prometheus:
    summary: true             # Include Prometheus metrics summary
    per_stage: false          # Disable Prometheus stage breakdown
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
    std_dev: 10         # standard deviation of the length of the synthetic prompts
    total_count: 100    # total number of prompts to generate to fit the above mentioned distribution constraints
  output_distribution:
    min: 10             # min length of the output to be generated
    max: 100            # max length of the output to be generated
    mean: 50            # mean length of the output to be generated
    std_dev: 10         # standard deviation of the length of the output to be generated
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
  prometheus:
    summary: true
    per_stage: true
```

### To Run Inference Perf Offline

```yaml
load:
  type: constant
  stages:
  - rate: 1
    duration: 30
api:
  type: chat
server:
  type: vllm
  model_name: ./models/SmolLM2-135M-Instruct
  base_url: http://0.0.0.0:8000
  ignore_eos: true
tokenizer:
  pretrained_model_name_or_path: ./models/SmolLM2-135M-Instruct
data:
  type: shareGPT
  path: ./data/shareGPT/ShareGPT_V3_unfiltered_cleaned_split.json # path to the downloaded shareGPT dataset
metrics:
  type: prometheus
  prometheus:
    url: http://localhost:9090
    scrape_interval: 15
report:
  request_lifecycle:
    summary: true
    per_stage: true
    per_request: false
  prometheus:
    summary: true
    per_stage: true
```

## Advanced Use Cases

### OpenTelemetry Trace Replay

Replay real-world LLM workloads captured as OpenTelemetry traces. This feature enables benchmarking with production traffic patterns, including complex dependency graphs, multi-turn conversations, and agent workflows.

#### Overview

OTel trace replay reconstructs the original call graph from trace files, preserving:
- **Sequential dependencies** — requests that must wait for predecessors
- **Parallel fan-outs** — concurrent requests with no dependencies
- **Shared-prefix patterns** — requests sharing common message history (KV-cache opportunities)
- **Output-aware replay** — substitutes recorded assistant messages with actual generated text for realistic growing-context behavior

#### How It Works

1. **Trace → Replay Graph**: Each trace file is converted to a directed acyclic graph (DAG) where:
   - LLM spans become nodes
   - Dependencies are inferred from message content (assistant messages matching predecessor outputs)
   - Timing gaps between calls are preserved as `wait_ms` delays

2. **Session-Based Execution**: Each trace file represents one *session*. The load generator controls:
   - How many sessions run concurrently (`concurrent_sessions`)
   - How many sessions to process per stage (`num_sessions`)
   - Optional rate limiting for session starts (`session_rate`)

3. **Output Substitution**: When a request depends on a predecessor's output, the recorded assistant message is replaced at runtime with the actual generated text, ensuring realistic KV-cache behavior for multi-turn conversations and agent chains.

#### Configuration

```yaml
data:
  type: otel_trace_replay
  otel_trace_replay:
    # Source — specify one:
    trace_files:                                  # List of specific trace files
      - "path/to/trace1.json"
      - "path/to/trace2.json"
    trace_directory: "path/to/traces/"            # OR: all .json files in directory

    # Model configuration
    use_static_model: true                        # Override recorded model names
    static_model_name: "my-model"                 # Model to use for all requests
    model_mapping:                                # OR: remap per recorded name
      "gpt-4": "my-model"
      "gpt-3.5-turbo": "my-other-model"

    # Generation parameters
    default_max_tokens: 1000                      # Fallback if output tokens are set to 0 in the otel file

    # Error handling
    include_errors: false                         # Skip spans with error status, that is, status != 0 (default)
    skip_invalid_files: true                      # Skip unparseable trace files during replay

load:
  type: trace_session_replay                      # Required for otel_trace_replay
  stages:
    - concurrent_sessions: 4                      # Max sessions active simultaneously
      num_sessions: 20                            # Run 20 sessions in this stage
      session_rate: 2.0                           # Optional: start max 2 sessions/sec
      timeout: 300                                # Optional: stage timeout in seconds
  num_workers: 4                                  # Worker processes
  worker_max_concurrency: 10                      # Max concurrent requests per worker
```

#### Stage Configuration

**`concurrent_sessions`** (required): Controls session-level concurrency
- `0` = unlimited (all sessions active at once, stress test mode)
- `N > 0` = at most N sessions active; when one completes, the next starts

**`num_sessions`** (optional): Number of sessions to run in this stage
- If omitted, runs all remaining sessions in the corpus
- Stages advance through the corpus sequentially (like standard load stages)

**`session_rate`** (optional): Rate limit for starting new sessions
- Omit for no rate limiting
- Useful for controlled ramp-up scenarios

**`timeout`** (optional): Wall-clock safety limit
- If exceeded, in-flight sessions are cancelled and stage exits as FAILED

#### Trace File Format

Traces must be JSON files with a `spans` array. Each LLM span requires:

```json
{
  "span_id": "unique-id",
  "trace_id": "trace-id",
  "start_time": "2024-01-01T00:00:00Z",
  "end_time": "2024-01-01T00:00:01Z",
  "name": "chat gpt-4",
  "attributes": {
    "gen_ai.request.model": "gpt-4",
    "gen_ai.input.messages": "[{\"role\":\"user\",\"content\":\"hello\"}]",
    "gen_ai.output.text": "hi there",
    "gen_ai.usage.prompt_tokens": 10,
    "gen_ai.usage.completion_tokens": 5
  }
}
```

Token counts are read from `gen_ai.usage.prompt_tokens` / `gen_ai.usage.completion_tokens` (also accepts `input_tokens` / `output_tokens`). If absent, a 4 chars/token estimate is used.

#### Example Configurations

**Simple Sequential Replay:**
```yaml
data:
  type: otel_trace_replay
  otel_trace_replay:
    trace_files:
      - "examples/otel/test_traces/simple/simple_chain.json"
    use_static_model: true
    static_model_name: "HuggingFaceTB/SmolLM2-135M-Instruct"
    default_max_tokens: 100

load:
  type: trace_session_replay
  stages:
    - concurrent_sessions: 1  # One session at a time
  num_workers: 4
  worker_max_concurrency: 10

server:
  type: vllm
  base_url: "http://localhost:8000"
  model_name: "HuggingFaceTB/SmolLM2-135M-Instruct"
```

**Multi-Stage with Rate Limiting:**
```yaml
data:
  type: otel_trace_replay
  otel_trace_replay:
    trace_directory: "examples/otel/test_traces/advanced"
    use_static_model: true
    static_model_name: "my-model"

load:
  type: trace_session_replay
  stages:
    - concurrent_sessions: 2
      num_sessions: 10
      session_rate: 1.0      # Warm-up: 2 concurrent, 1/sec start rate
    - concurrent_sessions: 5
      num_sessions: 20
      session_rate: 2.0      # Ramp-up: 5 concurrent, 2/sec start rate
    - concurrent_sessions: 10
                             # Final stage: 10 concurrent, all remaining sessions
  num_workers: 8
  worker_max_concurrency: 20
```

#### Use Cases

- **Production Traffic Replay**: Benchmark with real user interaction patterns
- **Agent Workflow Testing**: Replay complex multi-step agent traces with tool calls
- **Multi-Turn Conversation Analysis**: Test KV-cache efficiency with realistic conversation flows
- **Dependency Graph Validation**: Verify server behavior under complex request dependencies
- **Comparative Analysis**: Replay the same traces against different model configurations

#### Architecture Notes

Unlike standard data generators that produce independent requests, OTel trace replay operates at the session granularity. Each session is a complete trace file with an internal dependency graph. The load generator:

1. Activates sessions according to `concurrent_sessions` limit
2. Dispatches all events for a session immediately (workers handle internal parallelism)
3. Each event blocks until its predecessors complete and outputs are available
4. Tracks session completion and starts new sessions as slots become available

This design preserves the causal structure of the original workload while allowing the load generator to control session-level concurrency and throughput.