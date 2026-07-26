# Synthetic agent session examples

These configs exercise the `synthetic_agentic` data generator, which procedurally
builds multi-turn, tool-calling agent sessions (including recursive sub-agent fan-out) without
requiring a recorded OTel trace. Each config isolates one shape of the generator's behavior:
`single_agent_smoke` is a plain tool-call loop with no fan-out; `fanout_smoke` forces recursive
sub-agent spawning (`fanout_probability: 1.0`, `max_depth: 2`) to exercise the `tool_output`
merge path; `interactive_multiround` adds multiple human turns with think-time gaps;
`big_catalog` advertises a large tool catalog with no tool calls made, stressing prompt-side
token budgeting; and `mixed_theme` combines two themes with probabilistic fan-out and variable
tool-turn counts. All five have been run live against a local vLLM (or mock) server plus Jaeger
with zero replay failures.

To run one, start Jaeger and a target server (see `examples/otel/run_with_jaeger.sh` for the
Jaeger prerequisites), then:

```bash
./examples/otel/run_with_jaeger.sh examples/synthetic_agentic/configs/fanout_smoke.yml
```

To verify a run, reconstruct the sessions from Jaeger spans and check that every session
succeeded (no dangling `tool_call_id` / 400) and that the event counts match the config's
shape:

```bash
python tools/verify_synthetic_via_jaeger.py --expect-sessions 5 --min-events 7
```

Use `--expect-events N` instead of `--min-events` for configs with a deterministic per-session
event count (e.g. `single_agent_smoke`, `big_catalog`); use `--min-events` for configs with
fan-out or variable rounds, where the exact count depends on random draws.
