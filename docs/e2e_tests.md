# End-to-end tests: a contributor guide

This guide covers the end-to-end (e2e) test tier: the suites that run
`inference-perf` as a whole against a running server and assert on the reports
it produces. It explains how to run each suite locally and how to add a new
test. For the unit-level suites and the live-tier harness internals, see
[tests/README.md](../tests/README.md).

## The tiers at a glance

| Suite | Server | Where it runs | What it checks |
|---|---|---|---|
| Unit tests (`tests/`) | none: mocks and fakes | every push, merge-gated | component logic in isolation |
| Sim e2e (`e2e/tests/`, everything except `test_vllm_*`) | [llm-d-inference-sim](https://github.com/llm-d/llm-d-inference-sim), plus an ephemeral Prometheus where needed | every push, the `E2E Test on change` workflow, merge-gated | full-pipeline wiring: config in, reports out, metrics collection, exact-count golden token accounting, induced failure modes |
| Live-oracle slice (`e2e/tests/test_vllm_*.py`) | a real vLLM server in CPU mode | same workflow, one pass per release listed in `e2e/vllm_releases.txt`, merge-gated | server-sourced oracles: the real tokenizer, `usage` accounting, `/metrics` exposition |
| Live tier (`tests/optional/`) | real model servers on a GPU-backed Kubernetes cluster | hand-run before merging risky changes; scheduled runs are planned (#641, #643) | everything "passes against the sim" cannot prove: real weights, real accelerators, real serving stacks |

**The placement principle, from the v0.7.0 test matrix (#606): fake the
conditions, never the oracle.** A test that *induces* a condition (a stream
that breaks mid-flight, a mid-run outage) belongs against the sim or a
controllable fake, which can produce that condition on cue. A test whose
*oracle* is the server's own numbers (token counts from `usage`, metric names
on `/metrics`) cannot be faked without becoming tautological, so it takes the
cheapest real server that provides that oracle. CPU-mode vLLM exists in this
tier for exactly that reason: CPU mode changes generation speed, not token
accounting, so the real tokenizer and `/metrics` surface become CI-viable on a
plain runner.

## Running the sim-backed suite locally

The Nix dev shell provides everything: `llm-d-inference-sim`, `prometheus`,
`pdm`, and Python. Without Nix, install the sim by following its README; tests
that need a binary you do not have will skip, not fail.

```sh
nix develop
pdm run test:e2e
```

`test:e2e` runs `pytest e2e tests/optional -n auto --dist loadgroup`. The
live-tier cases under `tests/optional` auto-skip unless you pass
`--kubeconfigs`, so this is safe to run with no cluster. Tests run in parallel
under pytest-xdist; each test binds free ports for its own sim and Prometheus,
so concurrent copies do not clash.

## Running the live-oracle slice (CPU-mode vLLM)

The slice is every `e2e/tests/test_vllm_*.py` module. Selection is by naming
convention: a new `test_vllm_*.py` file joins the slice, and CI, with no
registration anywhere.

```sh
# Start a real vLLM CPU server (the same script CI uses):
e2e/vllm_cpu_server.sh

# In another shell:
pdm run test:e2e:live
```

Server resolution, in order:

1. `$E2E_VLLM_BASE_URL`, if set. CI sets this: the `E2E Test on change` job
   starts a `vllm/vllm-openai-cpu` container for each release listed in
   `e2e/vllm_releases.txt` and runs the slice against each one.
2. A `vllm` executable on `PATH` (spawned by `e2e/utils/vllm_server.py`).
3. Neither: the slice skips, and the run behaves as if it did not exist.

The slice runs serially on purpose (`test:e2e:live` does not use xdist), and
under `test:e2e`'s parallel run all vLLM modules share one
`xdist_group(name="vllm-cpu-server")` so tests that difference the server's
cumulative `/metrics` counters get exclusive server access.

## Running the live tier

The live tier drives real model servers on a GPU cluster and is auto-skipped
without one:

```sh
pdm run test:e2e --kubeconfigs=/path/to/kubeconfig
```

Cases declare their hardware needs in their own manifests (`nodeAffinity`),
queue on scarce hardware via a file-based semaphore, and skip when no cluster
node satisfies them. [tests/README.md](../tests/README.md) documents the
harness, cross-provider GPU targeting, and the useful flags (`--image`,
`--sweep-orphan-namespaces`).

## Adding a new e2e test

**First, decide whether it is an e2e test at all.** If the behavior is
observable from a component in isolation, write a unit test under `tests/`;
the e2e tier is for behavior that only exists when the whole pipeline runs
against a server.

Then pick the suite by what the test needs:

- **Needs real weights, accelerators, or a serving stack**: `tests/optional/`,
  as a case in an existing suite or a new suite directory.
- **Its oracle is a real server's own numbers** (tokenizer behavior, `usage`,
  `/metrics` names): a new or existing `e2e/tests/test_vllm_*.py` module. The
  naming convention is the registration.
- **Everything else**, including every induced-failure scenario: the sim suite
  under `e2e/tests/`.

The harness pattern, used by every existing test:

1. **Start the server** with the matching helper from `e2e/utils/`:
   `LLMDInferenceSimRunner` (the sim), `VLLMServerRunner` (CPU vLLM), or
   `GoldenSimServer` in `golden_sim.py` (a scripted server that returns
   byte-exact responses, for tests that need full control of the wire). The
   `prometheus_server` fixture in `e2e/conftest.py` provides an ephemeral
   Prometheus wired to a port your server should bind.
2. **Run the benchmark** with `run_benchmark_minimal` from
   `e2e/utils/benchmark.py`: it materializes your config dict to YAML, runs
   the real `inference-perf` CLI, and hands back parsed reports.
3. **Assert on the reports**, attaching the shared token-accounting helpers
   from `e2e/utils/accuracy.py`: `assert_successful_run` (refuses vacuous
   runs), `assert_output_token_accounting`, and `assert_streaming_bookkeeping`
   (chunk/ITL consistency). Use them even when your test targets something
   else; they are the invariant net that catches accounting regressions in
   passing runs.

House rules:

- **Guard on availability.** Skip (`pytest.mark.skipif`) when the binary or
  server your test needs is absent, so every environment can run the suite.
- **Bind free ports** via `e2e/utils/net.py`; never hardcode one.
- **Config lives with the test**, either inline as a dict passed to
  `run_benchmark_minimal` or under `e2e/configs/`.

## Lanes and gating

The v0.7.0 test matrix (#606) assigns every suite a lane by cost: per-change
work runs on every push with a hard time budget, at-merge work blocks the
merge but stays out of the push loop, and nightly work runs on a schedule and
never gates a PR. Today the whole e2e job runs on change and the live tier is
hand-run; moving each suite into its lane is CI-enforcement work tracked on
#606 (#642 required checks, #670 install caching, #641 and #643 for the live
lane). When lanes and this guide disagree, #606 is the source of truth.
