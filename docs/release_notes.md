# Release notes

## Unreleased

### Output accounting and streaming timing (#903)

- Chat Completions, Anthropic Messages, and both session replay paths count text,
  reasoning, and tool arguments in client output length. Tool names, IDs, and
  template wrappers are excluded. Server usage remains separate.
- Chat replay no longer substitutes server `completion_tokens` for client
  `output_len`. Client-based throughput and TPOT denominators can change even
  when server usage is present. `use_server_output_tokens` continues to select
  available server counts, with client fallback when usage is absent.
- Tool-only streamed turns now contribute to TTFT, TPOT, and ITL when their
  timelines meet each metric's requirements. TTFO remains answer text only.
- Timing changes also affect plain-text workloads, including one-token
  classification and buffered responses: a single arrival now retains TTFT;
  TPOT and ITL require at least two distinct arrival times and otherwise remain
  absent instead of reporting zero. This applies to native and benchmark-format
  reports.
- Tool-turn timeline/server disagreements have a separate mismatch counter with
  a 25% relative tolerance. This is a diagnostic threshold, not evidence of an
  accounting bug: template overhead can exceed it for short or empty calls.

See the [metric contract](metrics.md#tool-call-output-accounting) for counting,
serialization, and timing details. Historical reports may not be directly
comparable across these changes.
