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
    num_dataset_entries: 500
    use_static_model: true
    static_model_name: "mock-model"
    default_block_size: 64
    skip_invalid_files: true
    trace_idle_gap_cap_seconds: 1.0 # Caps think-time delay between turns to 1s
    # datagen_workers: 16 # Processes used to build sessions at startup.
                          # Defaults to available CPU cores; set to 1 for serial.
```

### Bounding a stage by time instead of session count

These corpora are large and usually are not replayed in full — the useful measurement is sustained load over a fixed window. Set `duration` on the stage instead of `num_sessions`:

```yaml
load:
  type: trace_session_replay
  stages:
    - concurrent_sessions: 16
      duration: 1800     # hold 16 conversations open for 30 minutes
      timeout: 2100      # safety net; must be longer than duration
```

At the deadline the stage stops dispatching and reports `COMPLETED` (not `FAILED`, which is what using `timeout` alone would give you). Sessions still running are recorded as **truncated**: counted under `num_sessions_truncated` and kept out of the session success/failure counts and duration percentiles, while the requests they completed still count.

The corpus has to be large enough to fill the window, since each session is drawn once. If it runs out early the stage ends short and logs a warning; raise `duplicate_sessions_target` until the corpus covers the window. See [OTel Trace Replay](otel_trace_replay.md#bounding-a-stage-by-time) for the full description.

---

## 🏃 Running the Benchmark

Run the benchmark with the following command:

```bash
python3 inference_perf/main.py -c configs/weka_trace_replay.yaml
```

The execution results will be written to the `reports-...` directory and summarized in [benchmark-results.md](../benchmark-results.md).
