# Conversation Replay

Benchmark agentic, multi-turn workloads where N conversations run concurrently and each
conversation's turns run sequentially as fast as the server will serve them. Conversations are
generated in memory from configurable token-length distributions (no trace files required) and
deterministically seeded for reproducible A/B comparisons.

## Table of Contents

- [When to use conversation replay](#when-to-use-conversation-replay)
- [Quick Start](#quick-start)
- [Configuration Guide](#configuration-guide)
- [Distribution types](#distribution-types)
- [Tool call latency simulation](#tool-call-latency-simulation)
- [How it fits together](#how-it-fits-together)

## When to use conversation replay

The question this data generator is built to answer is:

> *How many concurrent multi-turn conversations can my deployment sustain?*

Use this when you want to benchmark tool-calling / agent-style LLM workloads and:

- Conversations have many turns (tens per conversation) and accumulating context
- You want concurrency-bound dispatch (N in flight, fill as slots free), not a fixed QPS
- You want synthetic, parameterized input — no recorded traces
- You want reproducibility (same seed → identical input)

If instead you have recorded OpenTelemetry traces and want to replay their dependency graph
(parallel tool calls, branches, fan-outs), use [OTel Trace Replay](./otel_trace_replay.md). If
you want single-turn random prompts at a fixed QPS, use `shared_prefix` or `synthetic`.

## Quick Start

```yaml
load:
  type: concurrent
  stages:
    - num_requests: 2000
      concurrency_level: 5
    - num_requests: 2000
      concurrency_level: 10
    - num_requests: 2000
      concurrency_level: 20

api:
  type: completion

server:
  type: vllm
  model_name: Qwen/Qwen3-8B
  base_url: http://0.0.0.0:8000
  ignore_eos: true

tokenizer:
  pretrained_model_name_or_path: Qwen/Qwen3-8B

data:
  type: conversation_replay
  conversation_replay:
    seed: 42
    num_conversations: 200
    shared_system_prompt_len: 8359
    dynamic_system_prompt_len:
      type: normal
      min: 5000
      max: 45000
      mean: 14000
      std_dev: 15000
    turns_per_conversation:
      type: normal
      min: 25
      max: 57
      mean: 38
      std_dev: 6
    input_tokens_per_turn:
      type: lognormal
      min: 64
      max: 24000
      mean: 1652
      std_dev: 3000
    output_tokens_per_turn:
      type: lognormal
      min: 62
      max: 3600
      mean: 585
      std_dev: 400
```

The `concurrent` load type plus `preferred_worker_id` ensures each conversation is pinned to a
worker so its turns run strictly in order.

## Configuration Guide

All fields live under `data.conversation_replay`:

| Field | Type | Description |
| :--- | :--- | :--- |
| `seed` | int | RNG seed. Same seed → identical conversations, prompts, and turn lengths. |
| `num_conversations` | int | Number of conversation blueprints to pre-generate. Each conversation maps to one `LocalUserSession`; concurrency is driven by the `concurrent` load type. |
| `shared_system_prompt_len` | int | Tokens in the fixed prefix shared across all conversations. Useful for exercising prefix caching. |
| `dynamic_system_prompt_len` | [`Distribution`](#distribution-types) | Per-conversation suffix appended to the shared prefix. Makes each conversation's system prompt unique. |
| `turns_per_conversation` | [`Distribution`](#distribution-types) | How many user turns each conversation will issue before being recycled. |
| `input_tokens_per_turn` | [`Distribution`](#distribution-types) | User-message token length per turn. |
| `output_tokens_per_turn` | [`Distribution`](#distribution-types) | Target `max_tokens` per assistant turn. |
| `tool_call_latency_sec` | [`Distribution`](#distribution-types) | Optional: simulated tool-call delay between turns. See below. |

Any distribution field may be omitted; sensible per-field defaults are used when absent.

## Distribution types

Distribution fields use the shared [`Distribution`](../inference_perf/config.py) config, so the
same `type`, `min`, `max`, `mean`, `std_dev` schema works here and in other data generators.
Supported `type` values:

- `normal` — truncated normal clamped to `[min, max]`
- `lognormal` — heavy right tail; good match for real token-length distributions
- `uniform` — flat over `[min, max]`
- `fixed` — constant value (uses `mean`; `std_dev` is ignored)
- `skew_normal`, `poisson` — also supported for parity with other datagens

All sampling is seeded by the top-level `seed`, so runs are fully reproducible.

## Tool call latency simulation

Set `tool_call_latency_sec` to model the time an agent spends *between* LLM calls — typically
waiting on tool execution (DB lookup, API call, etc.):

```yaml
tool_call_latency_sec:
  type: lognormal
  min: 1
  max: 30
  mean: 8
  std_dev: 6
```

After each assistant response, the conversation sleeps for the sampled number of seconds
before releasing its session lock and starting the next turn. The sleep is async: the GPU
remains free to serve other concurrent conversations during the wait, so this correctly
models offline agentic workloads without artificially lowering throughput.

Omit `tool_call_latency_sec` (or set it to a `fixed` distribution with `mean: 0`) to measure
pure back-to-back GPU throughput.

## How it fits together

- **Data gen**: `ConversationReplayDataGenerator` pre-generates `num_conversations`
  blueprints at startup. Each blueprint owns a `LocalUserSession` (registered in the shared
  session registry) that enforces turn-by-turn ordering.
- **Load type**: use `load.type: concurrent`. Conversations are dispatched round-robin and
  pinned to workers via `preferred_worker_id`, so a given conversation's turns always run on
  the same worker and in order.
- **Recycling**: when a conversation exhausts its turns, the slot resets (fresh session,
  same blueprint) and replays from the beginning — letting a stage run as long as
  `num_requests` demands without running out of work.
- **Safety reset**: if a session's accumulated context exceeds ~2.7 M characters (≈225 K
  tokens, near typical `max_model_len` limits), the session is reset rather than letting the
  server reject the request.
