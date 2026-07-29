# Synthetic Agentic Session Replay

Generate agentic LLM workloads procedurally, without a recorded trace. `synthetic_agentic` builds
replay-graph sessions — multi-turn conversations, tool-calling loops, and recursive sub-agent
fan-out — from a handful of config knobs, then drives them against the target inference server
using the same session-replay runtime as [OTel trace replay](./otel_trace_replay.md).

Where OTel replay needs traces captured from a real system, `synthetic_agentic` needs only a
config. Use it to shape agentic load on demand: dial the tool-loop depth, fan-out breadth, tool
catalog size, and context growth to stress prefill / KV-cache the way real agent traffic does.

## Table of Contents

- [Why use synthetic agentic?](#why-use-synthetic-agentic)
- [Quick Start](#quick-start)
- [Configuration knobs](#configuration-knobs)
- [Examples](#examples)
- [How it works](#how-it-works)

## Why use synthetic agentic?

OTel trace replay is the right tool when you have real traces and want fidelity to them. But you
often want agentic load *before* you have traces, or load you can shape at will:

- Sweep a single dimension (e.g. tool-loop depth 0 → 8, or catalog size 8 → 500) to find where a
  server degrades, without hunting for traces that happen to have that shape.
- Reproduce a class of behavior (recursive fan-out, long accumulating context, wide parallel tool
  calls) deterministically, so a run is byte-identical given the same `(config, seed)`.
- Model context evolution — a session that grows and then **compacts** (a sharp prefill drop and
  KV-prefix reset) — which recorded traces rarely capture cleanly.

The generator produces the same `ReplayGraph` structure as OTel replay (a DAG of LLM calls with
sequential dependencies, parallel branches, and shared-prefix growth), so everything downstream —
session scheduling, output substitution, session-level metrics — is identical.

## Quick Start

```bash
# Run a demo config against a local vLLM server (+ Jaeger for tracing)
./examples/otel/run_with_jaeger.sh examples/synthetic_agentic/demo/03_single_agent_tool_loop.yml

# Or run directly
python -m inference_perf.main \
  --config examples/synthetic_agentic/demo/03_single_agent_tool_loop.yml

# Inspect a generated session graph without a server
# (this offline tool sizes turns with a real tokenizer, so the config needs a
#  top-level `tokenizer: {pretrained_model_name_or_path: "<model>"}` block)
python -m inference_perf.datagen.synthetic_agentic_to_replay_graph \
  --config examples/synthetic_agentic/demo/09_recursive_fanout.yml \
  --session-index 0 \
  --output /tmp/graph.json \
  --summary
```

A minimal config needs only the three required knobs; everything else has a sensible default:

```yaml
api:
  type: chat
  streaming: false
data:
  type: synthetic_agentic
  synthetic_agentic:
    num_sessions: 100
    input_tokens_per_turn:  {type: lognormal, mean: 600, std_dev: 400}
    output_tokens_per_turn: {type: uniform, min: 40, max: 200}
load:
  type: trace_session_replay
  stages: [{concurrent_sessions: 8}]
server:
  type: vllm
  base_url: "http://localhost:8000"
  model_name: "HuggingFaceTB/SmolLM2-135M-Instruct"
```

> **Like OTel replay,** `data.type: synthetic_agentic` **requires** `load.type: trace_session_replay`.

## Configuration knobs

All live under `data.synthetic_agentic`. Most sizing knobs are **distributions** (`{type: fixed |
uniform | lognormal | normal, ...}`), sampled once per session (or per agent) so different sessions
vary. Vocabulary: a *session* is one root agent; a *turn* is one user message to the root; within a
turn the agent runs a *tool loop* (call a tool, read the result, repeat) then answers, or spawns
*sub-agents* that each run their own loop and report back (recursively, down to `max_depth`).

| Knob | What it does | Default |
|---|---|---|
| `num_sessions` | Number of sessions to generate (load volume). | **required** |
| `input_tokens_per_turn` | Size of each new user turn's content. | **required** |
| `output_tokens_per_turn` | Size of a plain-text answer. (Tool-*call* outputs are sized from the call the generator builds, not this knob.) | **required** |
| `turns_per_session` | User turns to the root agent (1 = autonomous, N = interactive multi-turn; context accumulates across turns). | fixed 1 |
| `fanout_probability` | Chance an agent spawns sub-agents instead of just answering (rolled per root turn and per sub-agent). 0 = single-agent, 1 = full tree to `max_depth`. | 0 |
| `theme_mix` | Content theme(s) + weights: `generic`, `db2_latency_incident`, `research_rag`, `code_change_task`. | equal mix of all four |
| `tool_loop_depth` | How many times an agent goes around its tool loop before answering. 0 = answer directly; total model calls = this + 1. | fixed 2 |
| `parallel_tool_calls_per_step` | Tool calls emitted in a single response (one loop iteration's width). | fixed 1 |
| `tool_catalog_size_per_agent` | Number of tools advertised to the agent (prefill / KV stress). | fixed 8 |
| `sub_agents_per_spawn` | How many children a spawning agent creates. | uniform 2–4 |
| `max_depth` | Hard cap on sub-agent tree depth (a depth-`max_depth` agent never spawns). | 2 |
| `max_events_per_session` | Budget on LLM calls per session, so a dense fan-out can't explode. | 64 |
| `shared_system_prompt_len` | Tokens of a fixed system-prompt head that opens **every** agent call (the standing "system head" real agents carry). | 0 (none) |
| `context_compaction` | `{trigger_tokens, target_tokens}`: when a turn's accumulated input crosses `trigger_tokens`, the next turn starts fresh with a `target_tokens`-sized summary replacing the transcript (a prefill drop + KV reset). | off (pure growth) |
| `tool_call_latency_sec` | Pause between an agent's steps, modelling tool round-trip time (an offline wait that frees the GPU). | fixed 1s |
| `user_think_time_sec` | Pause before each follow-up turn (turns 2..N), modelling user read/think time. | fixed 10s |
| `max_model_len` | Fail-fast ceiling: reject a config whose worst-case peak request (inputs + output) would exceed the model's context window, instead of 400-ing mid-run. | off |
| `seed` | Base seed for deterministic per-session generation. | 42 |

## Examples

**Multi-turn conversation with a heterogeneous tool loop and context compaction** (a single agent
answering follow-ups over many turns, like a coding assistant):

```yaml
data:
  type: synthetic_agentic
  synthetic_agentic:
    num_sessions: 100
    turns_per_session:  {type: uniform, min: 3, max: 8}
    theme_mix: {code_change_task: 1.0}
    tool_loop_depth: {type: uniform, min: 0, max: 4}     # some turns answer directly, some loop deep
    tool_catalog_size_per_agent: {type: fixed, mean: 20}
    input_tokens_per_turn:  {type: lognormal, mean: 2000, std_dev: 800}
    output_tokens_per_turn: {type: fixed, mean: 150}
    context_compaction:
      trigger_tokens: {type: fixed, mean: 8500}
      target_tokens:  {type: fixed, mean: 1500}
```

**Recursive multi-agent fan-out** (an orchestrator that spawns sub-agents, each running its own
tool loop, like a coding agent delegating subtasks):

```yaml
data:
  type: synthetic_agentic
  synthetic_agentic:
    num_sessions: 100
    fanout_probability: 0.5                          # ~half the agents delegate
    sub_agents_per_spawn: {type: uniform, min: 1, max: 3}
    max_depth: 2
    theme_mix: {generic: 1.0}
    tool_loop_depth: {type: fixed, mean: 2}
    input_tokens_per_turn:  {type: fixed, mean: 400}
    output_tokens_per_turn: {type: fixed, mean: 80}
```

More ready-to-run configs live in [`examples/synthetic_agentic/`](../examples/synthetic_agentic/)
— the `demo/` directory has one numbered config per shape (bare, tool loop, parallel calls,
orchestrator fan-out, recursive fan-out, big catalog, …).

## How it works

The generator emits a `ReplayGraph` per session — the same structure OTel replay builds from traces
— so it plugs into the shared session-replay runtime (`ReplayGraphSessionGeneratorBase`) unchanged.
Each graph event is one LLM call whose input is the accumulating transcript; the model's reply is
the event's output.

- **Tool loop**: the agent's turn emits a tool call (forced via `tool_choice`), the result is fed
  back, and the loop repeats `tool_loop_depth` times before a final plain-text answer.
- **Fan-out**: a spawning agent's reasoning turn emits N parallel `dispatch_agent` tool calls in one
  message (mirroring how real harnesses spawn sub-agents); each child runs its own loop and reports
  back, and a merge turn synthesizes the reports. Children recurse down to `max_depth`.
- **Determinism**: all randomness derives from `seed` and the session index, so a given
  `(config, session_index)` produces a byte-identical graph across runs.
- **Content themes** supply the domain flavor (an SRE latency incident, a RAG research task, a code
  change, …); filler is sized to the token knobs so prompts hit the requested sizes.

For the underlying replay mechanics (segment decomposition, output substitution, tool-call replay,
session-level metrics, failure handling), see the [OTel Trace Replay](./otel_trace_replay.md)
developer guide — synthetic sessions replay through exactly that path.
