# OTel Trace Replay: Agentic Workload Benchmarking

This document describes a proposed addition to `inference-perf` that enables benchmarking of **agentic workloads** by replaying LLM call graphs extracted from OpenTelemetry traces. The change adds a new data-generation path alongside the existing one, without modifying any existing behavior.

---

## Motivation

Existing load types (`CONSTANT`, `POISSON`, `CONCURRENT`) generate independent, single-turn requests. Agentic applications — tool-calling agents, multi-turn conversations, RAG pipelines — produce **chains of dependent LLM calls** where each call's input includes the output of one or more predecessors. Replaying these workloads requires:

1. A way to express **dependency graphs** between requests within a session.
2. A mechanism to **substitute real generated outputs** into dependent requests at replay time, so KV cache patterns faithfully reflect production traffic.
3. **Session-level concurrency control** separate from per-request concurrency.
4. **Session-level metrics** (success rate, duration, total tokens per session) alongside the existing per-request metrics.

---

## API Changes: Generator Class Hierarchy

### Before

```
DataGenerator (ABC)
```

`DataGenerator` combined common API validation logic with request-level generation. All generators had to inherit from it even when they didn't stream individual requests.

### After

```
BaseGenerator (ABC)
├── DataGenerator       — unchanged request-based generation
└── SessionGenerator      — new session-based generation for agentic workloads
```

**`BaseGenerator`** (`inference_perf/datagen/base.py`) is a new abstract base that holds only the shared initialization — API config validation, tokenizer storage, and the `is_prefered_worker_requested()` hook. Both `DataGenerator` and `SessionGenerator` extend it.

**`DataGenerator`** is unchanged from the outside. It retains all its existing abstract methods (`get_data`, `is_io_distribution_supported`, `is_shared_prefix_supported`) and is used by all existing load types.

**`SessionGenerator`** is a new ABC for session-based replay. It defines the contract LoadGen uses to drive agentic workloads:

| Method | Purpose |
|---|---|
| `get_session_count()` | Total sessions in the corpus |
| `get_session_info(index)` | Metadata (session_id, file_path, num_events) |
| `get_session_event_indices(index)` | Event indices belonging to a session |
| `get_session_events(index)` | Session events as `LazyLoadInferenceAPIData` |
| `activate_session(session_id)` | Called by LoadGen when starting a session; marks root nodes ready |
| `check_session_completed(session_id)` | Polled by LoadGen to detect session completion |
| `build_session_metric(...)` | Constructs a `SessionLifecycleMetric` after completion |
| `cleanup_session(session_id)` | Releases per-session state to prevent memory growth |

---

## New Implementation: `OTelTraceReplayDataGenerator`

**File:** `inference_perf/datagen/otel_trace_replay_datagen.py`

`OTelTraceReplayDataGenerator` implements `SessionGenerator`. It parses OTel JSON trace files, builds a dependency graph per session, and serves events to LoadGen in the correct order.

### Trace Parsing and Graph Construction

OTel traces contain LLM spans with timestamps and message content. The generator (using helpers in `otel_trace_to_replay_graph.py`) extracts all LLM calls from a trace file and infers dependencies between them by time-window analysis: if a call's input messages overlap with a predecessor's output, a dependency edge is added to the graph. The result is a `ReplayGraph` — a DAG of nodes, where each node is one LLM call and edges represent "must complete before" relationships.

### Output-Aware Replay

A key design goal is that replay is *faithful*: downstream calls see the **actual generated output** of their predecessors, not the static recorded text. This is important for KV cache reuse measurement — the cache must be populated with the real generated tokens.

Two orthogonal coordination problems exist in this system, handled by two dedicated objects:

**`NodeOutputRegistry`** — solves the intra-worker problem: coroutines on the same worker's event loop need to wait for a predecessor's output before building their request. The registry holds plain in-process dicts (`node_id → output text`, `node_id → input messages`) and one `asyncio.Event` per node. When a node completes, `record()` writes the output and fires the event, immediately unblocking any dependent coroutines. No IPC is involved — session-to-worker affinity (see below) guarantees that every node in a session runs on the same worker, so cross-process sharing is never needed. `NodeOutputRegistry` has no `Manager` dependency.

**`SessionSharedState`** — solves the cross-process problem: the main process drives the session pool loop and needs two facts that workers produce:
1. Which nodes have completed (to detect session completion).
2. Which sessions have failed (to skip dependent nodes and retire sessions early).

Both are stored in `Manager.dict()` instances inside `SessionSharedState`, which is the single place in the codebase where multiprocessing proxies are created. Workers call `shared_state.record_node_completed()` and `shared_state.mark_session_failed()`; the main process calls `shared_state.is_node_completed()` and `shared_state.is_session_failed()`.

**`OTelChatCompletionAPIData`** — a `ChatCompletionAPIData` subclass that carries the `node_id`, `predecessor_node_ids`, `input_segments`, and references to both `registry` and `shared_state`. Before the HTTP request is dispatched, `wait_for_predecessors_and_substitute()` is called:

1. Checks `shared_state` for pre-existing session failure (fast-path skip).
2. Awaits all predecessor nodes in parallel via `asyncio.gather` over `registry.require_async()`, suspending the coroutine without consuming OS threads.
3. Checks `shared_state` again for session failure after waiting (a predecessor may have failed while waiting).
4. Substitutes `output` and `shared` segments in the message list with actual predecessor outputs from the registry.

After the HTTP response is processed, `on_completion()` registers the output in `registry` (unblocking dependents on the same worker) and records the completion in `shared_state` (making it visible to the main process) — in a single call, with no callback indirection.

**`SessionGraphState`** — tracks graph traversal per session (ready, dispatched, completed node sets). `check_session_completed()` reads from `shared_state.node_completions` to sync completed nodes into the local graph state, returning `True` when all nodes are accounted for.

### Session-to-Worker Affinity

All events of a session are routed to the same worker process. This is set by assigning `prefered_worker_id` on each `LazyLoadInferenceAPIData` in `get_session_events()`. Worker affinity means:
- `NodeOutputRegistry` can use plain dicts with no IPC overhead.
- `asyncio.Event` waiting works correctly (all coroutines of a session share the same event loop).
- Output substitution always finds predecessor outputs in the local cache.
- `SessionSharedState` only needs to carry completion signals and failure flags — not actual output data.

### Failure Handling

If an LLM call fails (HTTP error), `process_failure()` calls `shared_state.mark_session_failed()`. Any dependent nodes check `shared_state.is_session_failed()` at the pre- and post-wait points in `wait_for_predecessors_and_substitute()`, set `skip_request = True`, and call `on_completion()` with an empty output — propagating the failure through the graph without issuing further HTTP requests. The main process detects the failed session via the same `shared_state` when it polls `check_session_completed()`.

### Configuration

```yaml
data:
  type: otel_trace_replay
  otel_trace_replay:
    trace_dir: examples/otel/test_traces/
    model_map:
      "gpt-4o": "meta-llama/Llama-3.1-8B-Instruct"
    include_errors: false
    skip_invalid_files: true
```

---

## Load Generator: `run_session_stage`

**File:** `inference_perf/loadgen/load_generator.py`

A new method `run_session_stage()` handles `TRACE_SESSION_REPLAY` load stages. Standard stages (`run_stage`) remain unchanged.

### Session Pool Loop

`run_session_stage` maintains a **session pool** of bounded concurrency. The main loop:

1. **Fills the pool** up to `concurrent_sessions` by calling `dispatch_session()` for each new session.
2. **Rate-limits** session starts if `session_rate` (sessions/sec) is configured.
3. **Polls** active sessions by calling `datagen.check_session_completed(session_id)`. Completed sessions are retired: their `SessionLifecycleMetric` is built and recorded, and `cleanup_session()` is called to release memory.
4. Repeats until all sessions in the stage's slice have completed (or timeout is reached).

### Session Dispatch

`dispatch_session()` calls `datagen.activate_session()` to mark the session active (setting root nodes as ready in the graph state), then puts all of the session's events into the request queue via `request_queue.put(..., worker_id)` — routing to the session's preferred worker. The events are `LazyLoadInferenceAPIData` wrappers; actual request loading and predecessor waiting happen inside the worker.

### Worker-Side Execution

Workers already poll the request queue. For OTel replay, before calling `client.process_request()`, the worker now calls `await request_data.wait_for_predecessors_and_substitute()`. This is a no-op for non-OTel requests (the method is absent), so the existing path is unaffected. After dispatch, if the client returns an error, `process_failure()` is called on the request data so the session can be marked failed and dependents can skip.

### Multi-Stage Cursor

`LoadGenerator` tracks `_session_cursor` across stages. Each stage consumes a slice of sessions from the corpus, controlled by `stage.num_sessions`. This allows multi-stage experiments (e.g., warm-up + measurement) over the same session corpus without overlap.

### New Load Stage Config

```yaml
load:
  base_seed: 42
  stages:
    - type: trace_session_replay
      concurrent_sessions: 10
      session_rate: 2.0        # sessions/sec (optional)
      num_sessions: 100        # sessions to run in this stage (optional)
      timeout: 300             # per-stage timeout in seconds (optional)
  num_workers: 4
  worker_max_concurrency: 100  # set high — see note below
```

> **Note on `worker_max_concurrency`:** all events for a session are enqueued to the worker
> immediately when the session starts, even if most nodes are waiting on predecessors. Each
> waiting node holds a worker semaphore slot for the duration of its wait. Since waiting is
> done via `asyncio.Event` (zero threads — just a suspended coroutine), the cost of a high
> value is negligible. A safe rule of thumb:
> `worker_max_concurrency ≥ concurrent_sessions × avg_nodes_per_session`.

---

## Session-Level Reporting

### `SessionLifecycleMetric`

A new metric type (`inference_perf/apis/base.py`) captures the outcome of a full agentic session:

| Field | Description |
|---|---|
| `session_id` | Unique session identifier |
| `stage_id` | Stage that ran this session |
| `file_path` | Source trace file |
| `start_time`, `end_time`, `duration_sec` | Wall-clock timing |
| `num_nodes` | Total LLM calls in the session |
| `num_nodes_completed` | Calls that actually completed |
| `success` | `True` if all nodes completed without error |
| `error` | First error encountered, if any |
| `total_input_tokens`, `total_output_tokens` | Aggregated from request-level metrics |

### `SessionMetricsCollector`

`inference_perf/metrics/session_collector.py` decouples LoadGen (which produces session metrics) from ReportGen (which consumes them). LoadGen calls `collector.record_metric()` as sessions complete. After the run, `collector.enrich_metrics(request_metrics)` aggregates token counts and error status from the per-request metrics into each session metric. ReportGen then reads the enriched metrics via `collector.get_metrics()`.

### Reports

`ReportGenerator.generate_session_reports()` produces three report artifacts per run:
- **Summary** — aggregate statistics across all sessions (count, success rate, mean/p50/p90/p99 duration and token counts).
- **Per-stage breakdown** — same statistics grouped by stage.
- **Per-session detail** — one row per session with all fields.

---

## Backwards Compatibility

All changes are additive. The new `SessionGenerator` path is only activated when `data.type: otel_trace_replay` is set. Existing data generators, load types, and reports are unmodified. The `DataGenerator` hierarchy is unchanged from the outside — `BaseGenerator` extraction is an internal refactor with no impact on subclasses.
