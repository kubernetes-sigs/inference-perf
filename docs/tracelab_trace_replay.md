# TraceLab Trace Replay

The TraceLab Trace Replay capability benchmarks GenAI model servers by replaying real-world coding-agent
traces from the University of Washington SyFI Lab (Claude Code / Codex). Each trace row is one LLM invocation
with serving-relevant token accounting (`input_tokens_total = prefix_tokens + newly_append_tokens`), tool-call
metadata, and round ordering. It converts sessions into a **dependency graph of events** and replays them
concurrently, maintaining causal/dependency fidelity.

Dataset + paper: https://tracelab.cs.washington.edu / https://arxiv.org/abs/2606.30560

> Note: the public TraceLab trace is sanitized — raw prompts, completions, and tool payloads are removed.
> Each round's prompt is reconstructed as a single user message holding the whole prompt: the leading
> `prefix_tokens` of the previous round's text are reused and the rest is fresh synthetic text, sized so
> the message tokenizes to exactly `input_tokens_total` (cache misses share nothing; compacting rounds
> shrink). Expected outputs are sized to the recorded `output_tokens`. Inter-round spacing comes from
> `timing_events` timestamps when present, else the round's tool latency.
>
> Tokenizer: reconstruction uses the run's configured `tokenizer`, so prompt sizes are measured with
> the same tokenizer the benchmark runs against — point it at the target model's tokenizer for
> faithful prefill sizes.

---

## 🏗️ How it Works

1. **Trace Load**: rows are loaded from local TraceLab JSONL files (`.json`, `.jsonl`, `.jsonl.gz`) or a
   Hugging Face dataset (e.g. `UW-SyFI/TraceLab`), grouped by `session_id`, and ordered by `round_index`.
   For Hugging Face, the `rounds` table is joined with the `tools` and `timing_events` tables on
   `round_pk`, and `num_dataset_entries` caps sessions (not rows).
2. **Graph Compilation**: each round becomes one LLM call in a linear session chain. The transcript grows
   round-over-round (prior outputs are carried as assistant messages), so the shared replay-graph runtime
   infers causal edges and replays with output substitution like OTel/Weka replay.
3. **Session-based Execution**: a thread pool runs sessions concurrently. Within each session, events execute
   as soon as their predecessors complete.
4. **Think-Time Simulation**: the gap between the previous round's last `timing_events` timestamp and
   this round's first is replayed as think time (falling back to the previous round's tool latency when
   timestamps are absent), capped by `trace_idle_gap_cap_seconds`, or skipped with `ignore_trace_delays`
   (same knob shape as Weka replay).

---

## ⚙️ Configuration

```yaml
load:
  type: trace_session_replay
  stages:
    - concurrent_sessions: 16
      num_sessions: 100
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
  type: tracelab_trace_replay
  tracelab_trace_replay:
    trace_files:
      - ./traces/tracelab_round_trace.jsonl
    # trace_directory: ./traces/tracelab/
    # hf_dataset_path: "UW-SyFI/TraceLab"
    # num_dataset_entries: 100 # Max sessions from HuggingFace
    # filter: "lambda x: x['max_tokens'] < 262144" # Derived per session: max_tokens
    #                       (largest single-round input+output), total_tokens,
    #                       num_rounds. Uses eval(), so treat as trusted input.
    use_static_model: true
    static_model_name: "mock-model"
    skip_invalid_files: true
    trace_idle_gap_cap_seconds: 1.0 # Caps think-time delay between rounds to 1s
    # ignore_trace_delays: true # Run rounds back-to-back
    # disable_output_substitution: true # Replay built prompts verbatim instead of
    #   substituting live predecessor outputs (dependency timing still enforced)

report:
  request_lifecycle:
    summary: true
    per_stage: true
    per_request: true
```

See the [Reporting section of config.md](config.md#reporting) for `per_request_fields` options (recommended to
drop raw payloads for large-prompt replay runs, as in the Weka doc).
