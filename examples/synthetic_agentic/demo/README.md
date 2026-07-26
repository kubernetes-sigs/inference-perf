# Synthetic Agent Sessions — Demo Progression

A guided tour of the `synthetic_agentic` data generator, from the simplest shape to a
realistic mixed workload. Each config isolates one new concept, then later ones combine them.

Run any demo (with a vLLM server on `localhost:8000`):

```bash
python -m inference_perf.main --config examples/synthetic_agentic/demo/01_bare_no_tools.yml
# or with Jaeger tracing to inspect the per-session event graph:
./examples/otel/run_with_jaeger.sh examples/synthetic_agentic/demo/01_bare_no_tools.yml
```

**Model note:** every config is set to `HuggingFaceTB/SmolLM2-135M-Instruct` (~8K context) so demos
01–11 run on a small server today. Swap `server.model_name` / `base_url` for your real model when
ready. **Demo 12 needs a large-context model (≥128K)** — it will exceed SmolLM2's context and 400.

Demos 01–05 build up the **event model** one growth mechanism at a time (start here): a session is a
DAG of LLM calls where each call's INPUT is the cumulative transcript so far and the assistant reply
is that call's OUTPUT (not a separate event). Watch how the per-call input message count grows.

| # | Config | Demonstrates | What to look for |
|---|--------|--------------|------------------|
| 01 | `01_bare_no_tools` | Bare non-agentic baseline (no tools) | 1 event/session (the answer is the call's OUTPUT, not a separate event); no tools in requests |
| 02 | `02_multiturn_no_tools` | Multi-turn conversation, NO tools — pure conversation growth | 3 events; input msg count grows 1→3→5 across rounds; prompt tokens increase per round; no tools |
| 03 | `03_single_agent_tool_loop` | A tool-loop (introduces tools) — tool-loop growth | 3 events (principal + 2 tool turns; last turn's OUTPUT is the answer); input grows 1→3→5; catalog advertised |
| 04 | `04_long_tool_loop` | Long tool-loop (`tool_turns_per_loop`=5) — growth made obvious | 6 events; input msg count grows 1→3→5→7→9→11; prompt tokens climb every turn |
| 05 | `05_multiturn_with_tools` | Multi-turn AND tool-loops (both growth mechanisms combined) | 6 events; input msg count grows 1→3→3→5→5→7; prompt tokens INCREASE across rounds |
| 06 | `06_parallel_tool_calls` | K>1 tool calls per turn (best-effort) | 3 tool_calls + 3 matching results per turn |
| 07 | `07_chat_with_tools_varying` | Per-round tool usage varies | mix of answer-only and tool-using rounds |
| 08 | `08_orchestrator_fanout` | Spawn 2 sub-agents in parallel, wait, continue | 2 sub-agent calls overlap in time; merge is terminal (its OUTPUT is the answer) |
| 09 | `09_recursive_fanout` | Recursive fan-out (depth 2) | nested parallel sub-agents; no dangling ids |
| 10 | `10_big_catalog` | Tool-catalog inflation (prefill stress) | 1 event but large prompt; 30 tools advertised, none called |
| 11 | `11_mixed_everything` | Mixed themes + probabilistic fan-out + varying rounds | wide session-size spread; both themes; 0 errors |
| 12 | `12_realistic_scale` | Real Exgentic scale (~114K-token prompts, 487 tools) | **needs big-context model**; input tokens ~87K–136K |

**Verify a run** (reconstructs sessions from Jaeger spans and checks they match intent):

```bash
python tools/verify_synthetic_via_jaeger.py --expect-sessions 5 --expect-events 1   # 01 (bare, 1 event)
python tools/verify_synthetic_via_jaeger.py --expect-sessions 5 --expect-events 3   # 02/03 (grow 1→3→5)
python tools/verify_synthetic_via_jaeger.py --expect-sessions 5 --expect-events 6   # 04 (long tool-loop)
python tools/verify_synthetic_via_jaeger.py --expect-sessions 4 --min-peak-concurrency 2   # 08 (parallel sub-agents)
```

All demos are deterministic: the same config + `seed` produces byte-identical session graphs.
