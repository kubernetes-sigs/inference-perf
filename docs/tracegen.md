# Trace Generation

`inference-perf` includes built-in trace generation modules capable of generating complex workload patterns synthetically. Currently, it supports generating advanced shared prefix traces for Tree-of-Thought (ToT) patterns.

## General Usage

You can run trace generation as a standalone step before running a full benchmark by utilizing the `--tracegen` CLI flag.

```bash
inference-perf --tracegen <module>:<path/to/tracegen_config.yaml>
```

Presently, the supported module is:
- `tot`: Tree of Thought (ToT) synthetic trace generation.

---

## Tree of Thought (ToT) Trace Generation

The `tot` trace generation module simulates a Tree-of-Thought workload, where a language model explores multiple reasoning paths simultaneously (branches), creating a tree-like structure. The system maintains a set of active thoughts (beams or "expand top" nodes) and expands them layer by layer.

This trace generation produces a `SharedPrefix` formatted JSONL file. Each entry corresponds to an individual completion request in the tree. Because requests within the same branch share their entire generation history up to that point, they will carry identical `shared_prefix_id`s, instructing the benchmark runner to simulate prefix caching.

### Configuration Parameters

A typical ToT trace generation configuration file looks like this:

```yaml
# Example: configs/tracegen_tot.yaml
num_layers: 3
branching_factor: 3
expand_top: 2
system_prompt_len: 200
thought_prompt_len: 50
thought_len: 150
start_timestamp: 0.0
output_file: "data/trace_tot_cli.jsonl"
```

| Parameter            | Type    | Description |
|----------------------|---------|-------------|
| `num_layers`         | `int`   | The total number of thought layers (depth) to simulate in the tree. |
| `branching_factor`   | `int`   | The number of new thoughts (requests) to branch out from each active node. |
| `expand_top`         | `int`   | The number of nodes (thoughts) selected to advance to the next layer (akin to beam width). |
| `system_prompt_len`  | `int`   | The length of the initial system prompt (root node) in tokens. |
| `thought_prompt_len` | `int`   | The length of the new user prompt asking the model to expand the thought in tokens. |
| `thought_len`        | `int`   | The length of the model's generated thought response in tokens. |
| `start_timestamp`    | `float` | The initial starting timestamp in seconds. |
| `output_file`        | `str`   | The path where the generated JSONL trace file will be saved. |
| `graph_output`       | `str`   | (Optional) The path where a Graphviz `.dot` file of the generated tree will be saved. |

### Running the Trace Generator

To generate the trace, run the `inference-perf` CLI with the `--tracegen` flag:

```bash
inference-perf --tracegen tot:configs/tracegen_tot.yaml
```

The output will be written to `data/trace_tot_cli.jsonl` (as defined in the `output_file` parameter).

### Replaying the Trace

Once generated, the trace will follow the `SharedPrefix` trace format, which can be natively replayed by `inference-perf`. You can refer to this trace inside an `inference-perf` benchmark configuration.

```yaml
# Example: configs/trace-tot.yaml
load:
  type: trace_replay
  trace:
    file: data/trace_tot_cli.jsonl
    format: SharedPrefix
  worker_max_concurrency: 100

api:
  type: completion
  streaming: true

server:
  type: vllm
  model_name: google/gemma-3-1b-it
  base_url: http://0.0.0.0:8000
  ignore_eos: true

tokenizer:
  # The tokenizer is necessary to randomly generate token IDs
  pretrained_model_name_or_path: gpt2

data:
  type: shared_prefix
  trace:
    file: data/trace_tot_cli.jsonl
    format: SharedPrefix
```

You can then run the benchmark using the generated trace with the standard command:

```bash
inference-perf -c configs/trace-tot.yaml
```

### Trace Details

The generated trace follows the `SharedPrefix` JSONL format. Below is an example snippet of what the output looks like:

```json
{"timestamp":0.0,"shared_prefix_length":200,"tail_input_length":50,"output_length":150,"shared_prefix_id":1}
{"timestamp":0.01,"shared_prefix_length":200,"tail_input_length":50,"output_length":150,"shared_prefix_id":1}
{"timestamp":1.51,"shared_prefix_length":400,"tail_input_length":50,"output_length":150,"shared_prefix_id":2}
{"timestamp":1.52,"shared_prefix_length":400,"tail_input_length":50,"output_length":150,"shared_prefix_id":3}
```

- **Shared Contexts (`shared_prefix_id`)**: Requests with the same `shared_prefix_id` and `shared_prefix_length` will be given identical random prefix token distributions. In an LLM server, the first hit will compute the prefix representation and subsequent requests will enjoy a KV-cache hit.
- **Divergent Prompts (`tail_input_length`)**: The `tail_input_length` provides new token IDs appended uniquely to every individual request to represent the new portion of the reasoning sequence.
