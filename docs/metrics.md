# Inference Performance Metrics Definition

This document outlines the key metrics used for evaluating performance, their definition and how they are calculated.

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
| **Time to first token (TTFT)** | `time first non empty output token received - request send time` | ms | Calculating the time it takes for the user to receive the first token from the response
| **Time per output token (TPOT)** | `(e2e request latency - ttft ) / (output tokens - 1)` | ms per output token | Calculating the average time it takes for the user to receive successive tokens after the first token
| **Normalized time per output token** | `e2e request latency / output tokens` | ms per output token | Normalizing the request latency at the output token level for comparing different use cases
| **Inter Token Latency (ITL)** | `time between output token generation within a request` | ms per output token | Calculating the time it takes for the user to receive successive tokens after the first token, but at a more granular level than TPOT which averages the token latency within a request

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
| `output_len` | client | The response text re-tokenized as one whole message |
| `output_tokens` | server `usage.completion_tokens`, client `output_len` when the server reports none | Server-side this is an exact count of decode steps |
| `client_fallback_counts` | n/a | Per side (`prompt_tokens`, `output_tokens`), how many successful requests carry a client count because the server reported none. Nonzero means that distribution mixes sources |
| `token_count_mismatches` | n/a | Streamed requests where the sum of the per-chunk client tokenization differs from the server count |

Usage keys differ by API: OpenAI-compatible servers report `prompt_tokens` and
`completion_tokens`, the Anthropic Messages API reports `input_tokens` and `output_tokens`.
Both are read.

### Which count normalizes per-token latency

TPOT, normalized TPOT, output token throughput and token goodput divide by the client-side
`output_len` by default. Setting `report.request_lifecycle.use_server_output_tokens: true`
switches them to the server count for every request where the server reported one. The flag
does not change `output_len` or `output_tokens` themselves, only which of the two the
per-token metrics divide by.

### Reading a mismatch

A nonzero `token_count_mismatches` means client and server disagree on how many tokens the
response contained, so any metric normalized by the client count is off by that much. A
nonzero `client_fallback_counts` entry means the opposite problem: for those requests there is no
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

**Note:** vLLM's `cached_tokens` is block-granular — the reported value is quantized by the KV cache block size, so the hit rate may not reflect exact token-level precision.
