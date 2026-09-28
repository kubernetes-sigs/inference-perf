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

---

## ⏱️ Stage Timing

`trace_session_replay` stages (shared with OTel trace replay) support three independent
timing controls:

| Setting | Scope | Description |
|---------|-------|-------------|
| `stages[].duration` | Per stage | Optional planned stage length in seconds, used instead of `num_sessions`. The stage stops dispatching at the deadline and reports `COMPLETED` |
| `stages[].max_stage_duration` | Per stage | Optional wall-clock cap in seconds. Omit to run until every session in the stage completes. If exceeded, in-flight sessions are cancelled, never-started sessions are dropped, and the stage is marked failed/timed out. Stranded sessions are counted in the stage report as `sessions_not_completed_active` / `sessions_not_completed_pending` |
| `stage_teardown_grace_seconds` | Global (`load.`) | How long in-flight requests get to finish after a stage ends, for any reason, before being force-cancelled. Default `120.0`. Reported separately as `teardown_duration`, excluded from the stage's metrics window |

### Bounding a stage by time instead of session count

These corpora are large and usually are not replayed in full — the useful measurement is sustained load over a fixed window. Set `duration` on the stage instead of `num_sessions`:

```yaml
load:
  type: trace_session_replay
  stages:
    - concurrent_sessions: 16
      duration: 1800              # hold 16 conversations open for 30 minutes
      max_stage_duration: 2100    # safety net; must be longer than duration
  stage_teardown_grace_seconds: 30
```

At the deadline the stage stops dispatching and reports `COMPLETED` (not `FAILED`, which is what using `max_stage_duration` alone would give you). Sessions still running are recorded as **truncated**: counted under `num_sessions_not_completed_active` and kept out of the session success/failure counts and duration percentiles, while the requests they completed still count. Each one still reports its own event split — `num_events_completed` plus `num_events_cancelled` adds up to `num_events` — so a session cut off by the deadline is excluded from the aggregates without being unaccounted for.

**Unlike `otel_trace_replay`, this generator does not replay its corpus to fill the window.** It builds every session up front and holds them all in memory, so a session cannot be rebuilt once it has run and been released — replaying one would mean keeping the whole corpus alive for the whole run. The corpus therefore has to be large enough to fill the window, since each session is drawn once; if it runs out early the stage ends short and logs a warning saying by how much. Raise `duplicate_sessions_target` until the corpus covers the window. (That combination is rejected for `otel_trace_replay`, which replays instead — here it is the intended way to cover a window.)

See [OTel Trace Replay](otel_trace_replay.md#bounding-a-stage-by-time) for the rest of the description, and [OTel Trace Replay: Stage Timing](otel_trace_replay.md#stage-timing-max_stage_duration-and-stage_teardown_grace_seconds)
for a worked timeline — the config, load generator, and session report shape are identical
between the two datagens, except for corpus replay.

---

## 🏃 Running the Benchmark

Run the benchmark with the following command:

```bash
python3 inference_perf/main.py -c configs/weka_trace_replay.yaml
```

The execution results will be written to the `reports-...` directory and summarized in [benchmark-results.md](../benchmark-results.md).
