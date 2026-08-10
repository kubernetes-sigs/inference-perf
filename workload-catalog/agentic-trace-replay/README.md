# Agentic Trace Replay Workload

This workload replays real OpenTelemetry traces from agentic systems. Each session is a chain of causally dependent LLM calls — the agent calls the model, runs a tool on the result, feeds the result back, and repeats — with the original per-call inputs, outputs, and timing preserved.

## 1. Use Case and Distributions
**What it is**: Production agent traces — coding agents, browsing agents, customer-support flows — replayed against the inference server with the original sequence of calls, per-call inputs and outputs, and inter-call structure preserved.
**Observed characteristics** (from the reference dataset):
- **Input Sequence Length (ISL)**: Per-call inputs range from 0 to 3.4M tokens, with a mean of 49.1k tokens. The dataset includes diverse workloads from simple queries to complex multi-document contexts.
- **Output Sequence Length (OSL)**: Outputs are short, averaging 659 tokens (range: 0 to 146k tokens). Agents emit many short tool arguments and few long answers.
- **Number of Turns**: Sessions range from 1 to 300 calls per session, with a mean of 22 turns. The distribution is highly variable across different agent harnesses and benchmarks.
- **Shared Prefix**: 92.0% of calls share prefix with predecessors, averaging 44.6k shared-prefix tokens (45.1k cacheable) per call — massive KV cache hit opportunity. The remaining 8.0% are mostly first calls in sessions (4.57%) plus truly independent calls (3.40%).
- **Tool Usage**: 65.9% of sessions use tools, with 61.7% of all calls being tool calls. Tool definitions are present in 76.5% of sessions.
- **Session Duration**: Average 406 seconds (~6.8 minutes), with high variability (std dev: 998 seconds). Some sessions complete in under a second while others run over 2 hours.

> **Replay-time caveat**: the durations above are the ones recorded in the corpus. At replay
> time each inter-call gap is capped by `max_wait_ms` (default `15000`), so replayed sessions
> run shorter than the recorded mean unless you raise that cap. The cap keeps one pathological
> multi-hour tool call from stalling a benchmark run.

## 2. Reference Datasets
- **[Exgentic agent-llm-traces-v2](https://huggingface.co/datasets/Exgentic/agent-llm-traces-v2)**: 10,056 OpenTelemetry traces across six benchmarks (AppWorld, BrowseCompPlus, SWE-bench, TAU2 Airline/Retail/Telecom), five frontier models, and five agent harnesses (Claude Code, OpenAI solo, tool-calling, tool-calling with shortlisting, smolagents code).

> **Note**: The v2 dataset includes improved trace processing and normalization. The statistics in this workload are based on v2. For the legacy v1 dataset, use `Exgentic/agent-llm-traces`.

## 3. System Impact

What sets this apart from synthetic multi-turn workloads (`conversation_replay`) is that the *real* sequence of calls is replayed rather than a clean turn-by-turn dialogue. A synthetic generator assumes every call extends one monotonically growing conversation; real agent sessions do not always behave that way.

- **Causally Dependent Calls**: Each call's prompt contains the previous call's actual output (a tool result fed back in), so call *N* cannot start until call *N-1* finishes and its real output is substituted in. The chain length is set by the agent's control flow, not a fixed turn count — sessions run from 1 to 300 calls (mean: 22). This is the dependency structure that fixed-schedule load types cannot reproduce.
- **Conversation + Independent Calls in One Session**: Agents do not only carry one growing conversation. Some harnesses interleave the main dialogue with standalone, stateless calls — for example, every session of the `tool_calling_with_shortlisting` harness (300 sessions, AppWorld only) breaks off to issue short `[developer, user]` classification queries that carry none of the conversation history, then resumes the main thread. These two call shapes have completely different prefix-cache behavior, and a single session mixes both.
- **Growing Shared Context**: For the conversational calls, the prompt grows by the prior turn on every step (the same prefix, extended), so prefix caching and prefix-aware routing directly drive throughput while the KV footprint climbs across the session.
- **Tool-Call Overhead**: Calls that recorded a tool call are replayed with forced `tool_choice` and the original tool schemas, adding per-call constraint and validation cost that plain chat completions do not incur.

For a deeper look at the capture → replay-graph → output-aware-replay pipeline and how session-level metrics expose end-to-end workflow latency, see the [Benchmarking LLM Inference with Production Agent Traces](https://medium.com/inference-perf/benchmarking-llm-inference-with-production-agent-traces-f47f7f994aff) blog post.

## 4. Filtering and Scaling

Filter the dataset by benchmark, harness, or session size with the `filter` field in `data.otel_trace_replay` (a Python lambda applied to each record). Benchmarks differ widely in shape, so narrowing to one keeps a run homogeneous.

`max_tokens` on each record is the largest single call's input+output, so filtering on it
drops any session that would overflow the served context window. The cap must match the
context window your server actually serves, not the model's advertised maximum:
`Qwen/Qwen2.5-72B-Instruct` (the model in `inference-perf.yaml`) ships with
`max_position_embeddings: 32768` and no `rope_scaling`, so a default vLLM deployment serves
32k, and the 128k figure on the model card requires enabling YaRN explicitly. Check
`GET /v1/models` and read `max_model_len` to confirm what your endpoint serves.

Leave headroom for the completion on top of the input: tool-call events are replayed with
`max_tokens: 4096` (see `override_tool_call_max_tokens`), so a call with only ~30.5k input
tokens still overflows a 32k window. Hence the `< 28000` default below.

```yaml
data:
  type: otel_trace_replay
  otel_trace_replay:
    hf_dataset_path: Exgentic/agent-llm-traces-v2
    # Keep only sessions whose largest call fits the served context window.
    # 28000 suits a 32k window; raise to 120000 only if you serve 128k (YaRN enabled).
    filter: "lambda x: x.get('max_tokens', 0) < 28000"
    # Or narrow to a single benchmark:
    # filter: "lambda x: x['benchmark'] == 'tau2_retail'"
```

Coverage depends on this cap: `< 28000` keeps 5,632 of 10,056 sessions (56%), `< 120000`
keeps 9,176 (91%). Raising the cap past what the server serves does not buy coverage — it
turns those sessions into HTTP 400s, and since later calls depend on earlier ones, one
rejected call cancels the rest of its session.

To stress-test beyond the 10,056 available sessions, set `duplicate_sessions_target` to inflate the corpus. Duplicates are KV-cache-isolated automatically. See [OTel Trace Replay](../../docs/otel_trace_replay.md) for the full configuration reference.
