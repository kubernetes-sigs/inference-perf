# Synthetic Agent Sessions — Demo Progression

A guided tour of the `synthetic_agent_sessions` data generator, from the simplest shape to a
realistic mixed workload. Each config isolates one new concept, then later ones combine them.

Run any demo (with a vLLM server on `localhost:8000`):

```bash
python -m inference_perf.main --config examples/synthetic_agentic/demo/01_bare_no_tools.yml
# or with Jaeger tracing to inspect the per-session event graph:
./examples/otel/run_with_jaeger.sh examples/synthetic_agentic/demo/01_bare_no_tools.yml
```

**Model note:** every config is set to `HuggingFaceTB/SmolLM2-135M-Instruct` (~8K context) so demos
01–09 run on a small server today. Swap `server.model_name` / `base_url` for your real model when
ready. **Demo 10 needs a large-context model (≥128K)** — it will exceed SmolLM2's context and 400.

| # | Config | Demonstrates | What to look for |
|---|--------|--------------|------------------|
| 01 | `01_bare_no_tools` | Bare non-agentic baseline (no tools) | 1 event/session (the answer is the call's OUTPUT, not a separate event); no tools in requests |
| 02 | `02_single_agent_tool_loop` | A tool-loop (introduces tools) | 3 events (principal + 2 tool turns; the last turn's OUTPUT is the answer); input grows 1→3→5; catalog advertised |
| 03 | `03_parallel_tool_calls` | K>1 tool calls per turn (best-effort) | 3 tool_calls + 3 matching results per turn |
| 04 | `04_interactive_multiround` | Multi-round conversation, growing context | prompt tokens INCREASE across rounds; 6 events (3 rounds × (k=1 + 1)) |
| 05 | `05_chat_with_tools_varying` | Per-round tool usage varies | mix of answer-only and tool-using rounds |
| 06 | `06_orchestrator_fanout` | Spawn 2 sub-agents in parallel, wait, continue | 2 sub-agent calls overlap in time; merge is terminal (its OUTPUT is the answer) |
| 07 | `07_recursive_fanout` | Recursive fan-out (depth 2) | nested parallel sub-agents; no dangling ids |
| 08 | `08_big_catalog` | Tool-catalog inflation (prefill stress) | 1 event but large prompt; 30 tools advertised, none called |
| 09 | `09_mixed_everything` | Mixed themes + probabilistic fan-out + varying rounds | wide session-size spread; both themes; 0 errors |
| 10 | `10_realistic_scale` | Real Exgentic scale (~114K-token prompts, 487 tools) | **needs big-context model**; input tokens ~87K–136K |

**Verify a run** (reconstructs sessions from Jaeger spans and checks they match intent):

```bash
python tools/verify_synthetic_via_jaeger.py --expect-sessions 5           # 01/02/03
python tools/verify_synthetic_via_jaeger.py --expect-sessions 4 --min-peak-concurrency 2   # 06 (parallel sub-agents)
```

All demos are deterministic: the same config + `seed` produces byte-identical session graphs.
