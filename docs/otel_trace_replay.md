# OTel Trace Replay

Replay LLM workloads captured as OpenTelemetry traces. You bring traces collected from a real
system (e.g. an agent framework instrumented with OTel); inference-perf reconstructs the
original call graph — including sequential dependencies, parallel fan-outs, and shared-prefix
patterns — and drives those calls against the target inference server under test.

## Table of Contents

- [Why use OTel trace replay?](#why-use-otel-trace-replay)
- [Quick Start](#quick-start)
- [Configuration Guide](#configuration-guide)
- [Session-Level Metrics](#session-level-metrics)
- [OpenTelemetry Background](#opentelemetry-background)
- [Developer Guide](#developer-guide)

## Why use OTel trace replay?

Standard load types (`constant`, `poisson`, `concurrent`) dispatch requests at pre-scheduled times determined by the load timer. While they can handle sequential multi-turn conversations (via `shared_prefix` with user sessions), they cannot model **complex dependency graphs** where:
- Multiple LLM calls run in parallel (e.g., parallel tool calls, concurrent reasoning paths)
- Each call's input depends on outputs from multiple predecessors
- Timing between calls reflects real application logic (waiting for tool results, user input, etc.) rather than just clock-based scheduling

Agentic applications — tool-calling agents with parallel branches, multi-step RAG pipelines, complex workflows — produce these **dependency graphs** where the structure and timing of calls is determined by the application's control flow, not a fixed schedule.

OTel trace replay enables you to:
1. Benchmark **complex agentic workloads** with parallel execution and branching dependencies
2. Replay **production traffic patterns** with actual timing and dependency structures from real systems
3. Measure **KV cache effectiveness** with realistic shared-prefix and growing-context patterns
4. Test **session-level behavior** (success rates, end-to-end latency, failure propagation) for complete workflows

> **Note:** If you're unfamiliar with OpenTelemetry traces and spans, see the [OpenTelemetry Background](#opentelemetry-background) section.

## Quick Start

```bash
# Replay a single trace against a local vLLM server
python -m inference_perf.main \
  --config examples/otel/configs/per_case_config/simple_chain.yml

# Replay multiple traces from a directory
python -m inference_perf.main \
  --config examples/otel/configs/advanced/graph-replay.yml

# Inspect the replay graph for a trace (no server needed)
python -m inference_perf.datagen.otel_trace_to_replay_graph \
  --input  examples/otel/test_traces/simple/simple_chain.json \
  --output /tmp/graph.json \
  --summary
```

## Configuration Guide

### Basic Configuration

OTel trace replay requires two configuration sections: `data` (what to replay) and `load` (how to replay it).

```yaml
api:
  type: chat                           # Required: chat or completion
  streaming: true                      # Optional: enable streaming responses

server:
  type: vllm                           # Required: vllm, sglang, or tgi
  base_url: "http://localhost:8000"   # Required: inference server URL
  model_name: "HuggingFaceTB/SmolLM2-135M-Instruct"  # Required: model name

data:
  type: otel_trace_replay              # Required: activates trace replay mode
  otel_trace_replay:
    trace_directory: "path/to/traces/" # Required: source traces (or use trace_files)

load:
  type: trace_session_replay           # Required: must match data type
  stages:
    - concurrent_sessions: 4           # Required: max sessions running simultaneously
      num_sessions: 20                 # Optional: omit to run all remaining sessions
      session_rate: 2.0                # Optional: max new sessions/sec (omit for no limit)
  worker_max_concurrency: 500          # Optional: set high for trace replay (default: 100)
                                       # Rule of thumb: concurrent_sessions × 50-100
```

> **Important:** `data.type: otel_trace_replay` **requires** `load.type: trace_session_replay`. A validator enforces this at startup.
>
> **Note on `worker_max_concurrency`:** Set this high for trace replay. All events in a session are enqueued immediately, and events waiting for predecessors hold concurrency slots. However, waiting is done via `asyncio.Event` (zero threads—just suspended coroutines), so high values have negligible cost. **Rule of thumb:** `concurrent_sessions × 50` to `concurrent_sessions × 100` depending on your trace complexity.

### Data Configuration: `otel_trace_replay`

The `data.otel_trace_replay` section controls what traces to replay and how to process them.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `trace_files` | list[string] | One of `trace_files` or `trace_directory` | List of specific trace files. Supports glob patterns (e.g., `"path/*/*.json"`) |
| `trace_directory` | string | One of `trace_files` or `trace_directory` | Directory containing trace files. All `.json` files will be loaded |
| `use_static_model` | boolean | No (default: `false`) | Override all recorded model names with `static_model_name` |
| `static_model_name` | string | Required if `use_static_model: true` | Model name to use for all requests |
| `model_mapping` | dict | No | Map recorded model names to target models (e.g., `"gpt-4": "my-model"`) |
| `default_max_tokens` | integer | No (default: `1000`) | Fallback `max_tokens` for traces that don't specify it |
| `include_errors` | boolean | No (default: `false`) | Include spans marked as errors in the trace |
| `skip_invalid_files` | boolean | No (default: `true`) | Skip unparseable trace files instead of failing |

**Example:**

```yaml
data:
  type: otel_trace_replay
  otel_trace_replay:
    trace_directory: "production_traces/"
    use_static_model: true
    static_model_name: "llama-3-8b"
    default_max_tokens: 2048
    skip_invalid_files: true
```

### Load Configuration: `trace_session_replay`

The `load.trace_session_replay` section controls how sessions are executed. Unlike standard load types that dispatch requests independently, `trace_session_replay` operates on **sessions** where each trace file = one session containing multiple LLM calls with complex dependency graphs (including parallel branches and conditional paths).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `stages` | list | Yes | List of stage configurations (see below) |
| `worker_max_concurrency` | integer | No (default: `100`) | Max concurrent requests per worker. **For trace replay, set to `concurrent_sessions × 50-100`** since waiting events hold slots but use zero threads |

**Stage Configuration:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `concurrent_sessions` | integer | Yes | Max sessions running simultaneously. Set to `0` for unlimited (stress mode) |
| `num_sessions` | integer | No | Total sessions to run in this stage. Omit to run all remaining sessions (entire corpus if single stage) |
| `session_rate` | float | No | Optional rate limit for starting new sessions (sessions/sec) |

**Example:**

```yaml
load:
  type: trace_session_replay
  stages:
    # Stage 1: Warm-up with low concurrency
    - concurrent_sessions: 2
      num_sessions: 10
      session_rate: 1.0
    
    # Stage 2: Ramp up
    - concurrent_sessions: 4
      num_sessions: 20
      session_rate: 2.0
    
    # Stage 3: Stress test (unlimited concurrency)
    - concurrent_sessions: 0
      num_sessions: 50
  
  worker_max_concurrency: 200
```


### Complete Configuration Example

```yaml
api:
  type: chat
  streaming: true

server:
  type: vllm
  base_url: "http://localhost:8000"
  model_name: "llama-3-8b"

data:
  type: otel_trace_replay
  otel_trace_replay:
    trace_directory: "production_traces/"
    use_static_model: true
    static_model_name: "llama-3-8b"
    default_max_tokens: 2048
    skip_invalid_files: true

load:
  type: trace_session_replay
  stages:
    - concurrent_sessions: 2
      num_sessions: 10
      session_rate: 1.0
    - concurrent_sessions: 4
      num_sessions: 20
      session_rate: 2.0
  worker_max_concurrency: 200
```

## Session-Level Metrics

In addition to per-request metrics (TTFT, TPOT, throughput), OTel trace replay produces **session-level metrics** that capture the outcome of complete agentic workflows.

### SessionLifecycleMetric

Each session (one trace file) produces a metric with:

| Field | Description |
|-------|-------------|
| `session_id` | Unique session identifier |
| `stage_id` | Stage that ran this session |
| `file_path` | Source trace file |
| `start_time`, `end_time`, `duration_sec` | Wall-clock timing for the entire session |
| `num_events` | Total LLM calls in the session graph |
| `num_events_completed` | Calls that actually executed and returned a response |
| `num_events_cancelled` | Calls skipped because a predecessor failed |
| `success` | `True` if all events completed without error |
| `error` | First error encountered, if any |
| `total_input_tokens`, `total_output_tokens` | Aggregated across all calls in the session |

### Reports

After a run, three session report files are generated:

- **`summary_session_lifecycle_metrics.json`** — Aggregate statistics across all sessions:
  - `num_sessions`, `num_sessions_succeeded`, `num_sessions_failed`
  - `total_events`, `total_events_completed`, `total_events_cancelled`
  - Distributions: `session_duration_sec`, `num_events`, `total_input_tokens`, `total_output_tokens`
  
- **`stage_N_session_lifecycle_metrics.json`** — Same statistics grouped by stage

- **`per_session_lifecycle_metrics.json`** — One entry per session with all fields (for detailed analysis)

These complement the standard per-request metrics, giving you both micro (individual LLM calls) and macro (complete workflows) views of performance.

## OpenTelemetry Background

**OpenTelemetry (OTel)** is an observability framework for collecting traces, metrics, and logs from distributed systems. A **trace** represents a complete request flow through your system, composed of multiple **spans**.

### What is a Span?

A **span** represents a single unit of work or operation. In the context of LLM applications:
- Each LLM API call (e.g., a chat completion request) is captured as a span
- A span includes timing information (start/end), input/output data, and metadata
- Spans are linked together via parent-child relationships to form a trace

### What is a Trace?

A **trace** is a collection of spans that together represent a complete workflow. For example:
- A multi-turn conversation: user message → LLM response → user follow-up → LLM response
- An agentic workflow: initial query → tool call → tool result → final answer
- A RAG pipeline: query → retrieval → context injection → generation

Each trace has a unique `trace_id`, and all spans within that trace share this ID. Spans also have their own `span_id` and reference their parent span, forming a directed acyclic graph (DAG) of operations.

**Why this matters for replay:** OTel trace replay reconstructs these dependency relationships from your production traces, ensuring that benchmark workloads maintain the same causal dependencies, timing patterns, and context-sharing behavior as your real system.

---

## Developer Guide

### Trace File Format

Bring traces exported from any OTel-instrumented LLM system. Each file is a JSON object with a `spans` array. Each LLM span must include:

```jsonc
{
  "span_id": "abc123",
  "trace_id": "xyz",
  "start_time": "2024-01-01T00:00:00Z",
  "end_time":   "2024-01-01T00:00:01Z",
  "name": "chat gpt-4",
  "attributes": {
    "gen_ai.request.model": "gpt-4",
    "gen_ai.input.messages": "[{\"role\":\"user\",\"content\":\"hello\"}]",
    "gen_ai.output.messages": "[{\"role\":\"assistant\",\"content\":\"hi\"}]",
    "gen_ai.usage.prompt_tokens": 10,
    "gen_ai.usage.completion_tokens": 5
  }
}
```

The replayer follows the [OpenTelemetry Semantic Conventions for GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/).

### How It Works: Trace → Replay Graph

Each OTel trace file contains a flat list of spans. The replayer converts them into a directed acyclic graph (DAG) that preserves the original dependencies and timing:

1. **Extract LLM spans** — Spans with `gen_ai.input.messages` (or a `chat *` name) become session events
2. **Infer dependencies** — Two types of edges are added:
   - **Causal edges**: when a span's input contains an `assistant` message whose content exactly matches a predecessor's output
   - **Temporal edges**: to the closest non-overlapping earlier span (timing fallback)
3. **Transitive reduction** — Redundant edges are pruned so only direct predecessors remain
4. **Preserve timing** — The delay between when predecessors finish and when each call starts is recorded as `wait_ms`

Root events (no predecessors) start immediately. All others wait for their predecessors, then observe `wait_ms` before dispatching.

### Architecture Overview

OTel trace replay introduces a new generator hierarchy to handle causally dependent requests. The codebase has two distinct generator types, both inheriting from `BaseGenerator`:

**`DataGenerator`** — Used by standard load types (`random`, `shared_prefix`, `cnn_dailymail`)
- Implements `get_data()` iterator yielding independent requests
- Works with `load.type: constant`, `poisson`, or `concurrent`
- Requests are fully independent

**`SessionGenerator`** — Used exclusively for trace replay
- Implements session-oriented methods instead of `get_data()`
- Works with `load.type: trace_session_replay`
- Requests within a session are **causally dependent**

### Why SessionGenerator Exists

OTel trace replay cannot use the `DataGenerator` model because:
1. Requests inside a trace are **causally dependent** — call B cannot start until call A finishes
2. A's actual output must be injected into B's prompt (not the recorded text)
3. A flat iterator has no way to express "don't yield this yet" or "substitute with live output"

`OTelTraceReplayDataGenerator` works at the granularity of whole *sessions* (one trace file = one session).

### SessionGenerator API

| Method | Purpose |
|--------|---------|
| `get_session_count()` | Total sessions in the corpus |
| `get_session_info(index)` | Metadata (session_id, file_path, num_events) |
| `activate_session(session_id)` | Marks root events as ready to dispatch |
| `get_session_events(index)` | Returns all events for a session |
| `check_session_completed(session_id)` | Returns `True` when all events finished |
| `build_session_metric(...)` | Constructs a `SessionLifecycleMetric` |
| `cleanup_session(session_id)` | Releases per-session state |

All requests for a session are enqueued immediately (for parallelism), but each request only *executes* once its predecessors complete — signalled via `EventOutputRegistry` on the same worker.

### Segment Decomposition

Each event's input is split into message-level segments:

- **`shared`** — Leading messages identical to a predecessor (KV-cache hit opportunity)
- **`output`** — An assistant message whose content is a predecessor's output (substituted at replay time with the actual generated text)
- **`unique`** — Messages unique to this call

This decomposition happens during graph construction and enables:
1. Accurate simulation of KV-cache behavior (shared prefixes)
2. Dynamic output substitution (growing context patterns)
3. Realistic context growth in multi-turn conversations

### Output-Aware Replay Implementation

Three coordination mechanisms handle output substitution and dependency management:

#### EventOutputRegistry

Intra-worker only. Holds plain dicts (`event_id → output text`, `event_id → input messages`) and one `asyncio.Event` per session event.

- When an event completes, `record()` writes the output and fires the signal, immediately unblocking dependent coroutines on the same worker
- When an event fails, `record_failure()` fires the signal without writing any output; `require_async()` detects this and raises `EventFailedError`
- No IPC — session-to-worker affinity guarantees all events of a session run on the same worker

#### WorkerSessionTracker

Per-worker session state tracking. Each worker independently tracks which events have completed and which sessions have failed within its assigned sessions. No cross-process communication needed due to session-to-worker affinity.

#### session_completion_queue

Event-driven worker→main communication. When the last event of a session completes, the worker pushes a completion notification (with event completion times and failure status) to an `mp.Queue`. The main process consumes from this queue in `check_session_completed()` instead of polling shared state.

#### Request Flow

Each `SessionChatCompletionAPIData` holds references to `registry`, `worker_tracker`, and `completion_queue`:

1. Before dispatching an HTTP request, the worker calls `wait_for_predecessors_and_substitute()`
2. This awaits predecessors via `registry.require_async()` (zero threads — pure `asyncio.Event` suspension)
3. Checks `worker_tracker` for session failure before and after waiting
4. Substitutes output segments with actual predecessor text
5. After the response returns, `on_completion()` writes to `registry` (unblocking dependents) and `worker_tracker` (recording completion)
6. If this was the last event in the session, pushes to `completion_queue`

### Failure Handling Details

When an event fails (network error, timeout, HTTP error), the system ensures dependent events don't hang and the session completes gracefully:

**Worker-level failure handling:**
1. `process_failure()` is called on the failed event's `SessionChatCompletionAPIData`
2. The worker marks the entire session as failed in `WorkerSessionTracker` (local to that worker)
3. `registry.record_failure(event_id)` is called — this sets the event's `asyncio.Event` without writing any output to `EventOutputRegistry`, keeping the registry clean
4. Dependent events unblock, receive an `EventFailedError` from `require_async`, and skip without making HTTP requests

**Session-level failure propagation (within a worker):**
- **Pre-wait check**: Before waiting for predecessors, each event checks if its session has failed in `WorkerSessionTracker`. If so, it sets `skip_request = True`, calls `record_failure` on itself (to unblock its own successors), and returns immediately
- **Predecessor wait**: `asyncio.gather` awaits all predecessors via `require_async`. If any predecessor was marked failed, `require_async` raises `EventFailedError`. The event catches this, sets `skip_request = True`, calls `record_failure` on itself, and returns — propagating the failure hop-by-hop through the dependency graph
- **No empty outputs**: Cancelled events never write to `EventOutputRegistry`. The registry only contains real outputs from events that actually ran
- **No completion counting for skipped events**: Skipped events do not call `record_event_completed`. Session completion is signalled entirely via the immediate failure notification in `process_failure`
- **Session-to-worker affinity**: All events of a session run on the same worker, so `WorkerSessionTracker` (local to each worker) is sufficient for intra-session failure detection

**Worker-to-main-process communication:**
- On the first failure in a session, `process_failure` immediately pushes a completion notification to `session_completion_queue` with `"failed": True` and a `"cancelled_events"` count (how many events will be skipped as a result of this failure). This does not wait for skipped events to finish
- The main process calls `_process_completion_queue()` which sets `ReplaySessionState.is_complete` and `ReplaySessionState.failed` for the session
- When ending OTEL session spans, the load generator checks `ReplaySessionState.failed` to mark failed sessions with error messages

**Session metrics:**
- Session metrics include a `success` field (False for failed sessions) and an `error` field with the failure reason
- The `cancelled_events` field in the completion notification records how many events were skipped due to the failure (computed as `total_events − completed_before_failure − 1`)

This design ensures:
- No deadlocks: dependent events never wait indefinitely for failed predecessors
- Clean registry: no phantom empty-string entries for cancelled events
- Clean shutdown: sessions complete even when events fail, without waiting for all events to skip
- Accurate metrics: failures are tracked at both event and session level, with cancelled counts
- Accurate OTEL traces: failed sessions are marked with error messages in their spans
- Resource efficiency: failed sessions don't consume unnecessary worker time

**Note:** OTel trace replay always runs in multiprocess mode (requires `num_workers > 0`) because it uses `SessionGenerator`, which is not supported in single-process mode.

### Load Generator: run_stage vs run_session_stage

**`run_stage`** is the standard path used by every other load type:
1. Calls `get_data()` to produce a flat sequence of requests
2. Stamps each with a time from a `LoadTimer` (constant rate or Poisson)
3. Puts them all on the worker queue up front
4. Waits until `finished_requests_counter` reaches the expected total

This works because requests are independent and the load shape is fully determined before dispatch begins.

**`run_session_stage`** is the OTel-specific path. It cannot pre-compute a flat request list because the number of active requests at any moment depends on which sessions are in flight and how far each has progressed through its graph. Instead it runs a session pool loop:

1. Maintain a pool of at most `concurrent_sessions` active sessions
2. When the pool has room (and `session_rate` allows), pop the next session from the pending list, call `activate_session`, and enqueue all of its events at once
3. Poll each active session with `check_session_completed`; when one finishes, remove it from the pool so a new session can start
4. Exit when all sessions in this stage's corpus slice have completed

The key insight is that *session* concurrency (how many traces are in flight) is controlled here in the load generator, while *request* concurrency within a session is controlled by the dependency graph itself — root events run immediately, dependent events wait. The worker pool size (`num_workers` × `worker_max_concurrency`) sets the ceiling on how many LLM calls can be in flight across all sessions simultaneously.

### Dependency Inference Algorithm

The replayer infers dependencies using two types of edges:

1. **Causal edges**: When a span's input contains an `assistant` message whose content exactly matches a predecessor's output
2. **Temporal edges**: To the closest non-overlapping earlier span (timing fallback)

The temporal fallback is necessary because output matching doesn't always detect all dependencies. If event X ends before event Y begins, X is considered a predecessor even if Y doesn't use X's entire output.

After adding edges, **transitive reduction** prunes redundant edges so only direct predecessors remain.

### Backwards Compatibility

All changes are additive. The `SessionGenerator` path is only activated when `data.type: otel_trace_replay` is set. Existing data generators, load types, and reports are unmodified.
