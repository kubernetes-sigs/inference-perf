# Weka Trace Replay

The Weka Trace Replay capability allows you to benchmark GenAI model servers by replaying complex, real-world multi-agent execution traces. It converts a raw trace into a **dependency graph of events** (parent-child turns, tool calls, subagent spawns) and replays it concurrently, maintaining full causal/dependency fidelity.

---

## 🏗️ How it Works

1. **Dataset Download**: At startup, `inference-perf` downloads the Weka trace dataset from Hugging Face (e.g. `semianalysisai/cc-traces-weka-with-subagents-060826-256k`).
2. **Graph Compilation**: The replay generator parses each trace session, compiling individual turns and events into an **Execution Graph** of nodes. Each node represents an event (an inference call or tool/subagent execution). Traces are compiled in parallel across CPU cores (see `datagen_workers`); the output is deterministic and independent of the parallelism level.
3. **Causal Propagation**: Nodes register parent-child relationships. The text output of parent nodes is dynamically cached and substituted into the prompt messages of child nodes at runtime (e.g. tool execution output is placed back in the next LLM call).
4. **Session-based Execution**: A thread pool runs sessions concurrently. Within each session, nodes are executed as soon as their parents complete.
5. **Think-Time Simulation**: Think times and idle gaps between turns are simulated and capped using `trace_idle_gap_cap_seconds`.

---

## ⚙️ Configuration

To use Weka Trace Replay, define the data and load sections in your configuration YAML:

```yaml
load:
  type: trace_session_replay
  stages:
    - concurrent_sessions: 16
      num_sessions: 391
  num_workers: 8
  worker_max_concurrency: 100

api:
  type: chat
  streaming: true

server:
  type: mock # or openai/vllm/sglang/tgi

tokenizer:
  pretrained_model_name_or_path: HuggingFaceTB/SmolLM2-135M-Instruct

data:
  type: weka_trace_replay
  weka_trace_replay:
    hf_dataset_path: "semianalysisai/cc-traces-weka-with-subagents-060826-256k"
    num_dataset_entries: 500 # Caps accepted traces, not lines scanned
    # filter: "lambda x: x['max_tokens'] < 262144" # Keep traces that fit a context window.
                          # Derived per trace: max_tokens (largest single-request
                          # input+output), total_tokens, num_turns (flattened
                          # request count, so a subagent's three calls count as
                          # three). Uses eval(), so treat the expression as
                          # trusted input.
    use_static_model: true
    static_model_name: "mock-model"
    default_block_size: 64
    skip_invalid_files: true
    trace_idle_gap_cap_seconds: 1.0 # Caps think-time delay between turns to 1s
    # separate_subagent_session_ids: true # Give each child stream its own session ID.
    # datagen_workers: 16 # Processes used to build sessions at startup.
                          # Defaults to available CPU cores; set to 1 for serial.

report:
  request_lifecycle:
    summary: true
    per_stage: true
    per_request: true
    # Recommended for trace replay: Weka requests carry very large prompts, so
    # storing raw payloads can grow per_request_lifecycle_metrics.json to
    # multiple GB per run. Keep metadata and computed metrics only.
    per_request_fields:
      request: false          # Drop raw request payloads
      response: false         # Drop raw response payloads
      response_chunks: false  # Drop raw streaming chunks
      info: true              # Keep timestamps, token counts, server usage (cached_tokens)
      computed_metrics: true  # Keep per-request TTFT/TPOT/ITL
```

See the [Reporting section of config.md](config.md#reporting) for details on `per_request_fields`.

### Subagent Session IDs

By default, every request in a trace uses the enclosing replay session ID, including
subagent requests. Set `data.weka_trace_replay.separate_subagent_session_ids: true`
to give each subagent stream a separate request identity. Configure
`api.session_id_header_key` to send that identity to a session-affinity router:

```yaml
api:
  session_id_header_key: x-session-id
data:
  weka_trace_replay:
    separate_subagent_session_ids: true
```

Parent requests retain IDs such as `wekatrace0_example`. Child requests use
`<parent_id>::sa:<agent_id>:s<stream_index>`, for example
`wekatrace0_example::sa:researcher:s0`. The stream index starts at zero and is
present even when a subagent has only one stream. Sequential requests in a stream
share an ID; overlapping requests are packed into separate streams.

Duplicated replay sessions resolve child IDs from their own parent ID, for example
`wekatrace0_example_dup1::sa:researcher:s0`. The setting changes request headers and
session-token routing, while scheduling, dependency tracking, request metrics,
and session lifecycle reports still use the enclosing trace
session. The default is `false` to preserve existing benchmark behavior.

---

## ⏱️ Stage Timing

`trace_session_replay` stages (shared with OTel trace replay) support two independent
timing controls:

| Setting | Scope | Description |
|---------|-------|-------------|
| `stages[].max_stage_duration` | Per stage | Optional wall-clock cap in seconds. Omit to run until every session in the stage completes. If exceeded, in-flight sessions are cancelled, never-started sessions are dropped, and the stage is marked failed/timed out. Stranded sessions are counted in the stage report as `sessions_not_completed_active` / `sessions_not_completed_pending` |
| `stage_teardown_grace_seconds` | Global (`load.`) | How long in-flight requests get to finish after a stage ends, for any reason, before being force-cancelled. Default `120.0`. Reported separately as `teardown_duration`, excluded from the stage's metrics window |

```yaml
load:
  type: trace_session_replay
  stages:
    - concurrent_sessions: 16
      num_sessions: 391
      max_stage_duration: 600
  stage_teardown_grace_seconds: 30
```

See [OTel Trace Replay: Stage Timing](otel_trace_replay.md#stage-timing-max_stage_duration-and-stage_teardown_grace_seconds)
for the full explanation and a worked timeline — the config, load generator, and session
report shape are identical between the two datagens.

---

## 🏃 Running the Benchmark

Run the benchmark with the following command:

```bash
python3 inference_perf/main.py -c configs/weka_trace_replay.yaml
```

The execution results will be written to the `reports-...` directory and summarized in [benchmark-results.md](../benchmark-results.md).
