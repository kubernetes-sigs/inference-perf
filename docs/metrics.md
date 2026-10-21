# Inference Performance Metrics Definition

This document outlines the key metrics used for evaluating performance, their definition and how they are calculated.

For the Prometheus metrics inference-perf exports about its own runtime (as opposed to the benchmark result metrics defined here), see [runtime_metrics.md](./runtime_metrics.md).

## Throughput

| Metric | Formula | Unit | Used For
| :--- | :--- | :--- | :---
| **Output tokens / second** | `total output tokens / time in seconds` | tokens / second | Calculating output token throughput of the inference server
| **Input tokens / second** | `total input tokens / time in seconds` | tokens / second | Calculating input token throughput of the inference server
| **Requests / second** | `total requests completed / time in seconds` | qps | Calculating request throughput of the inference server

---

## Latency

| Metric | Formula | Unit | Used For
| :--- | :--- | :--- | :---
| **Time per request (e2e request latency)** | `request completion time - request send time` | seconds per request | Calculating how long a request takes to complete
| **Time to first token (TTFT)** | `time first non empty output token received - request send time` | ms | Calculating the time it takes the server to produce the first token of the response, reasoning and tool arguments included
| **Time to first output token (TTFO)** | `time first non empty content token received - request send time` | ms | Calculating the time it takes for the user to receive the first token of the answer, after any reasoning. Equals TTFT when answer text is the first generated channel; absent for tool-only or reasoning-only turns
| **Time per output token (TPOT)** | `(last token time - first token time) / (output tokens - 1)` | ms per output token | Calculating the average time it takes for the user to receive successive tokens after the first token
| **Normalized time per output token** | `e2e request latency / output tokens` | ms per output token | Normalizing the request latency at the output token level for comparing different use cases
| **Inter Token Latency (ITL)** | `time between output token generation within a request` | ms per output token | Calculating the time it takes for the user to receive successive tokens after the first token, but at a more granular level than TPOT which averages the token latency within a request

The ms units are what the CLI summary table shows. The JSON reports record every latency in seconds.

### Reasoning models

Reasoning models stream their reasoning on a separate channel (`reasoning` or
`reasoning_content` on OpenAI-compatible servers, `reasoning` when a server sends both,
`thinking` blocks on the Anthropic Messages API) before the answer. The server generates
and counts those tokens like any other, so TTFT, TPOT, ITL and output token counts include
them. TTFT is the time to the first generated token, which can be reasoning, text, or tool arguments.
TTFO is the time to the first token of the answer itself. The gap between TTFT and TTFO is the
time spent generating reasoning or tool arguments before the answer started.

A reasoning model served without a reasoning parser sends its reasoning inside the content,
usually wrapped in `<think>` tags. The reasoning then counts as content: TTFO equals TTFT.
Compare runs with and without the server's reasoning parser with that in mind.

---

## Price/Performance

| Metric | Formula | Unit | Used For
| :--- | :--- | :--- | :---
| **$ per million output tokens*** | `((accelerator $ / second) / (output tokens / second)) * 1M` | $ | Calculating the cost to serve million output tokens
| **$ per million input tokens*** | `((accelerator $ / second) / (input tokens / second)) * 1M` | $ | Calculating the cost to serve million input tokens
| **Throughput / $** | `(output tokens / second) / (accelerator $ / second)` | million output tokens | Calculating the performance to price ratio to get the throughput we are able to achieve for the cost spent

*\*Note: input and output token cost might need to be divided in mixed-batching cases since they are handled together by the server, using some factor like 1:4 for cost to generate input vs output tokens.*

---

## Token Accounting and Provenance

Every token count in a report comes from one of two places. The **server** reports `usage` on
the response; the **client** tokenizes the prompt it sent and the text it received. The two
disagree in practice, because of tokenizer revision differences, chat-template and tool-schema
overhead the client does not model, and streamed text re-encoded in fragments. The report
therefore keeps both and records which is which.

| Field | Source | Notes |
| :--- | :--- | :---
| `prompt_tokens` | server `usage.prompt_tokens`, client tokenization when the server reports none | Resolved per request while the response is processed. Supersedes `prompt_len`. |
| `prompt_tokens.cached` / `.uncached` | server `usage.prompt_tokens_details` | Absent when the server does not report the detail |
| `output_len` | client | Client re-tokenized text, reasoning, and tool-call arguments; each channel and each assembled call counted separately, without special tokens |
| `output_tokens` | server `usage.completion_tokens` / `usage.output_tokens`, client `output_len` when the server reports none | Server-side this is an exact count of decode steps |
| `client_fallback_requests` | n/a | Per side (`prompt`, `output`), how many successful requests carry a client count because the server reported none. Counts requests, not tokens. Nonzero means that distribution mixes sources |
| `token_count_mismatches` | n/a | Streamed turns without tool calls whose per-chunk client count differs from server usage |
| `tool_token_count_mismatches` | n/a | Streamed tool turns whose per-chunk client count differs from server usage by more than 25% of the larger count |

Usage keys differ by API: OpenAI-compatible servers report `prompt_tokens` and
`completion_tokens`, the Anthropic Messages API reports `input_tokens` and `output_tokens`.
Both are read.

### Which count normalizes per-token latency

TPOT, normalized TPOT, output token throughput and token goodput divide by the client-side
`output_len` by default. Setting `report.request_lifecycle.use_server_output_tokens: true`
switches them to the server count for every request where the server reported one. It resolves
that count from the same usage keys the report does, so either spelling switches the metrics.
The flag does not change `output_len` or `output_tokens` themselves, only which of the two the
per-token metrics divide by.

### Reading a mismatch

A nonzero `token_count_mismatches` means the client timeline count differs from server
usage on a non-tool turn. Tool turns use `tool_token_count_mismatches` and the tolerance
described below. Neither counter compares the assembled `output_len` directly: fragment
boundaries can change tokenization. A
nonzero `client_fallback_requests` entry means the opposite problem: for those requests there is no
server number to compare against, and `output_tokens` is carrying the client count. Both are
worth checking before comparing runs, and before comparing against another tool (see
[comparability.md](./comparability.md)).

The CLI summary's Token Length Aggregates table labels each column with its source and reports
any fallbacks in the table caption.

---

## Session-Based KV Cache Hit Rate

| Metric | Formula | Unit | Used For
| :--- | :--- | :--- | :---
| **kv_cache_hit_percent** | `100 × (total cached prompt tokens / total prompt tokens)` | percent (0–100) | Token-weighted cache hit rate across all sessions
| **kv_cache_hit_per_session_percent** | distribution of `100 × (session cached tokens / session prompt tokens)` | percent (0–100) | Per-session cache hit rate distribution (min/mean/max/percentiles)

Both metrics use the server-reported `prompt_tokens` as the denominator and `prompt_tokens_details.cached_tokens` as the numerator. Sessions where the server does not report cache information are excluded (reported as `None`, not 0%).

**Server requirement:** vLLM must be started with `--enable-prompt-tokens-details` for the server to populate `prompt_tokens_details.cached_tokens` in the usage response.

**Note on all-zero readings:** an all-zero `cached_tokens` run is ambiguous — it can mean a genuinely cold cache, or usage counters normalized to `0` by a gateway or sidecar in front of the engine (e.g. llm-d's prefill/decode sidecar, see [#818](https://github.com/kubernetes-sigs/inference-perf/issues/818)). The reported rate stays `0%` as the server sent it, and report validation flags it as a `session.cache` warning in `validation.json` suggesting verification against server-side cache metrics.

**Note:** vLLM's `cached_tokens` is block-granular — the reported value is quantized by the KV cache block size, so the hit rate may not reflect exact token-level precision.

### Tool-call output accounting

Chat Completions, Anthropic Messages, and both session replay paths count generated
text, reasoning, and tool arguments in `output_len`, independently of server usage.
Each call's argument fragments are assembled before counting; tool names, IDs,
protocol wrappers, and model-specific template tokens are excluded. OpenAI argument
strings retain their original spelling, including malformed or incomplete JSON.
Unary Anthropic input objects use `json.dumps(input, ensure_ascii=False)` with its
default separators. Streaming uses the received JSON fragments; the empty `{}` at
`content_block_start` is a placeholder, while a nonempty starting input is counted.
Serialization and whitespace differences can therefore change client counts between
unary and streaming responses. Server `usage` remains unchanged.

Tool-only deltas retain their arrival timestamps. TTFT starts at the first generated
text, reasoning, or argument token. TTFO starts only at answer text, so a tool-only
turn has no TTFO and arguments before text can make TTFT earlier than TTFO without
any reasoning. TPOT and ITL need at least two distinct arrival times; a whole call
received at once has TTFT but no TPOT or ITL. Tokens within a chunk share its timestamp.
The per-fragment token timeline can differ from the assembled `output_len` because
fragment boundaries need not coincide with tokenizer boundaries.

`token_count_mismatches` counts exact timeline/server-count disagreements for turns
without tool calls. `tool_token_count_mismatches` counts tool turns only when
`abs(client - server) > 0.25 * max(client, server)`, using the re-tokenized timeline
count and server output usage (`completion_tokens` or `output_tokens`). Missing usage
produces neither mismatch. The separate counter and tolerance keep template-token
and fragment-tokenization differences from drowning the text-turn signal.

Compared with earlier reports, agentic runs now include tool-only turns in TTFT,
TPOT, ITL, and output throughput (subject to the timing requirements above).
