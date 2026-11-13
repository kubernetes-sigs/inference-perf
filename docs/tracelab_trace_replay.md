# TraceLab Trace Replay

The TraceLab Trace Replay capability benchmarks GenAI model servers by replaying real-world coding-agent
traces from the University of Washington SyFI Lab (Claude Code / Codex). Each trace row is one LLM invocation
with serving-relevant token accounting (`input_tokens_total = prefix_tokens + newly_append_tokens`), tool-call
metadata, and round ordering. It converts sessions into a **dependency graph of events** and replays them
concurrently, maintaining causal/dependency fidelity.

Dataset + paper: https://tracelab.cs.washington.edu / https://arxiv.org/abs/2606.30560

> Note: the public TraceLab trace is sanitized — raw prompts, completions, and tool payloads are removed.
> Each round's prompt is rebuilt as a single user message holding the whole prompt: the leading 64-token
> blocks of the previous round's prompt that `prefix_tokens` covers are reused and the rest is fresh
> synthetic text, sized so the message tokenizes to exactly `input_tokens_total`, counting the special
> tokens the tokenizer adds (cache misses share nothing; compacting rounds shrink). Each request asks for
> the recorded `output_tokens`. A round that recorded zero output tokens asks for the client's default length.
> Inter-round spacing comes from `timing_events` timestamps when present, else the previous round's tool
> latency.
>
> KV-cache profile: every prompt is the offline-built one, so a server with prefix caching can reuse the
> leading blocks of the previous round. Prompt text is built from blocks named by id, and a block id is
> the same text everywhere in its session, so a round that reuses the previous round's leading blocks
> sends the same leading text whatever special tokens the tokenizer (gemma, Llama 3, Mistral) adds. Reuse
> is in whole blocks: up to 63 tokens at the end of a recorded prefix are sent as fresh text. A round with
> `prefix_tokens = 0` (cache miss) shares nothing and reprocesses the whole prompt, and a compacting
> round truncates the shared prefix so reuse stops at the divergence point. Live model outputs are never
> fed back into later prompts: unlike OTel/Weka replay, a round's response does not change its
> successors' inputs.
>
> Tokenizer: reconstruction uses the run's configured `tokenizer`, so prompt sizes are measured with
> the same tokenizer the benchmark runs against — point it at the target model's tokenizer for
> faithful prefill sizes. Prompt text comes from `data.corpus_file_path` (Shakespeare by default); a corpus
> smaller than the prompt is repeated as many times as needed, so long TraceLab rounds are never
> under-sized.

---

## 🏗️ How it Works

1. **Trace Load**: rows are loaded from local TraceLab JSONL files (`.json`, `.jsonl`, `.jsonl.gz`,
   `.json.gz`; stems may contain dots, e.g. `tracelab_v0.0.2.jsonl`) or a Hugging Face dataset (e.g.
   `UW-SyFI/TraceLab`), grouped by `session_id`, and ordered by `round_index`.
   For Hugging Face, the `rounds` table is joined with the `tool_calls` and `timing_events` tables on
   `round_pk`. Tables are **streamed** by default, so a small run never downloads and Arrow-builds the
   whole corpus (~665k rounds, ~744k tool calls, ~2.7M timing events) before the first row; set
   `hf_dataset_path: {path: "UW-SyFI/TraceLab", streaming: false}` for full local caching.
   `num_dataset_entries` caps sessions (not rows) for every source, and reading stops once that many
   sessions pass the filter. Only the `tool_calls` and `timing_events` rows belonging to kept sessions are
   indexed.

   Rows of one session must be **contiguous** in the source (the early stop depends on it). A `session_id`
   that reappears after another session is a hard error instead of silently replacing the rounds already
   read — sort the file by `session_id` or export one file per session.
2. **Graph Compilation**: each round becomes one LLM call in a linear session chain. The call's input is
   a single synthetic user message rebuilt from the recorded token accounting, and each round depends on
   the round before it, for ordering and timing only. Because that message is not a predecessor's
   assistant output, replay never substitutes live model output into a later prompt. Sessions are built
   when they are first run, so only the recorded numbers are held for the ones still waiting.
   Rounds are ordered by submission timestamp (tie-breaking on `round_index`, then `round_id`) when the
   trace carries timestamps, so duplicate `(session_id, round_index)` rows and rows whose index disagrees
   with their timestamps still replay in recorded order.
3. **Session-based Execution**: a thread pool runs sessions concurrently. Within each session, events execute
   as soon as their predecessors complete.
4. **Think-Time Simulation**: a round is *submitted* when its last input event arrives (`user_message`, or
   the last `tool_result` of several parallel tool calls) and *completes* at its last model-output event
   (`reasoning`, `text`, `tool_call`). `usage_report` (token_count) rows are written after the tool ran, so
   they are excluded from both bounds. The gap between the previous round's completion and this round's
   submission is replayed as think time. When timestamps are absent it falls back to the previous round's
   tool wall time, merging overlapping `[emitted_at, result_at]` intervals so concurrent tools are not
   over-counted. Setting `ignore_trace_delays: true` overrides the cap: every inter-round gap becomes zero
   and rounds run back-to-back.

   `trace_idle_gap_cap_seconds` (default 60s) caps each gap. Because the runtime clamps every wait to
   `max_wait_ms` (inherited default 15s), an unset `max_wait_ms` is raised to the cap; an explicit
   `max_wait_ms` is respected as-is. Set `trace_idle_gap_cap_seconds: null` (or `<= 0`) to replay recorded
   pauses verbatim — required for KV-cache TTL/eviction benchmarks over multi-minute human think time, where
   the runtime clamp is lifted instead of reinterpreting `max_wait_ms: 0` (which other replay types read as
   "no waiting at all").

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
    # hf_dataset_path: "UW-SyFI/TraceLab" # streamed by default
    # hf_dataset_path: {path: "UW-SyFI/TraceLab", streaming: false} # full local cache
    # num_dataset_entries: 100 # Max sessions, applied to every source
    # filter: "lambda x: x['max_tokens'] < 262144" # Derived per session: max_tokens
    #                       (largest single-round input+output), total_tokens,
    #                       num_rounds. Uses eval(), so treat as trusted input.
    use_static_model: true
    static_model_name: "mock-model"
    skip_invalid_files: true
    trace_idle_gap_cap_seconds: 1.0 # Caps think-time delay between rounds to 1s.
    #   null or <= 0 replays the recorded pauses verbatim (KV-cache TTL/eviction
    #   benchmarks); max_wait_ms, inherited at 15s, is raised to this cap unless
    #   you set max_wait_ms explicitly
    # max_wait_ms: 60000 # Runtime clamp on a single wait; default follows the cap
    # ignore_trace_delays: true # Run rounds back-to-back; overrides the gap cap
    #   and sets every inter-round gap to zero

report:
  request_lifecycle:
    summary: true
    per_stage: true
    per_request: true
```

See the [Reporting section of config.md](config.md#reporting) for `per_request_fields` options (recommended to
drop raw payloads for large-prompt replay runs, as in the Weka doc).

The format runs on the workload record layer. `data.type: workload_replay` with `format: TraceLab` reads the
same local files without this block: no Hugging Face source, filter, session cap or gap cap. See
[loadgen.md](loadgen.md#tracelab).
