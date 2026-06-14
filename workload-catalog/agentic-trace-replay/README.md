# Agentic Trace Replay Workload

This workload replays real OpenTelemetry traces from agentic systems. Each session is a chain of causally dependent LLM calls — the agent calls the model, runs a tool on the result, feeds the result back, and repeats — with the original per-call inputs, outputs, and timing preserved.

## 1. Use Case and Distributions
**What it is**: Production agent traces — coding agents, browsing agents, customer-support flows — replayed against the inference server with the original sequence of calls, per-call inputs and outputs, and inter-call structure preserved.
**Why the distributions** (observed in the reference dataset, not parameterized):
- **Input Sequence Length (ISL)**: Log-Normal. Per-call inputs range from ~9k tokens (TAU2 customer-service flows) to ~63k (AppWorld personal-assistant context).
- **Output Sequence Length (OSL)**: Log-Normal. Outputs are short, averaging 90–540 tokens; agents emit many short tool arguments and few long answers.
- **Number of Turns**: Log-Normal. Median 7 (TAU2 Airline) to 35 (SWE-bench), with a long tail up to 158 calls in a single session.

## 2. Reference Datasets
- **[Exgentic agent-llm-traces](https://huggingface.co/datasets/Exgentic/agent-llm-traces)**: 1,781 OpenTelemetry traces across six benchmarks (AppWorld, BrowseCompPlus, SWE-bench, TAU2 Airline/Retail/Telecom), five frontier models, and five agent harnesses (Claude Code, OpenAI solo, tool-calling, tool-calling with shortlisting, smolagents code).

> **Note**: When Exgentic publishes a newer revision of the dataset, update the `hf_dataset_path` in `inference-perf.yaml` (use the dict form with `revision:` to pin a specific version).

## 3. System Impact

What sets this apart from synthetic multi-turn workloads (`conversation_replay`) is that the *real* sequence of calls is replayed rather than a clean turn-by-turn dialogue. A synthetic generator assumes every call extends one monotonically growing conversation; real agent sessions do not always behave that way.

- **Causally Dependent Calls**: Each call's prompt contains the previous call's actual output (a tool result fed back in), so call *N* cannot start until call *N-1* finishes and its real output is substituted in. The chain length is set by the agent's control flow, not a fixed turn count — sessions run from 7 to 158 calls. This is the dependency structure that fixed-schedule load types cannot reproduce.
- **Conversation + Independent Calls in One Session**: Agents do not only carry one growing conversation. Some harnesses interleave the main dialogue with standalone, stateless calls — for example, `tool_calling_with_shortlisting` (every session in the corpus) breaks off to issue short `[developer, user]` classification queries that carry none of the conversation history, then resumes the main thread. These two call shapes have completely different prefix-cache behavior, and a single session mixes both.
- **Growing Shared Context**: For the conversational calls, the prompt grows by the prior turn on every step (the same prefix, extended), so prefix caching and prefix-aware routing directly drive throughput while the KV footprint climbs across the session.
- **Tool-Call Overhead**: Calls that recorded a tool call are replayed with forced `tool_choice` and the original tool schemas, adding per-call constraint and validation cost that plain chat completions do not incur.

## 4. Filtering and Scaling

Filter the dataset by benchmark, harness, or session size with the `filter` field in `data.otel_trace_replay` (a Python lambda applied to each record). Benchmarks differ widely in shape, so narrowing to one keeps a run homogeneous.

`max_tokens` on each record is the largest single call's input+output, so filtering on it
drops any session that would overflow the served context window. The cap must match the
model in `inference-perf.yaml`: Qwen3-8B serves 32k tokens natively and 128k only with YaRN
enabled. The example below assumes the 128k window and leaves headroom for the response;
lower the cap to `< 28000` when serving at the native 32k.

```yaml
data:
  type: otel_trace_replay
  otel_trace_replay:
    hf_dataset_path: Exgentic/agent-llm-traces
    # Keep only sessions whose largest call fits the served context window
    filter: "lambda x: x.get('max_tokens', 0) < 120000"
    # Or narrow to a single benchmark:
    # filter: "lambda x: x['benchmark'] == 'tau2_retail'"
```

To stress-test beyond the 1,781 available sessions, set `duplicate_sessions_target` to inflate the corpus. Duplicates are KV-cache-isolated automatically. See [OTel Trace Replay](../../docs/otel_trace_replay.md) for the full configuration reference.
