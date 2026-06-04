# Tests

```
tests/
├── required/    # Run in CI on every PR. Self-contained, no external deps.
├── optional/    # Live end-to-end tier: real model servers on a real cluster.
│   ├── harness/      # Shared, backend-agnostic plumbing (not a suite).
│   ├── multimodal/   # Suite: multimodal cases against a Qwen3-VL deployment.
│   └── text/         # Suite: a text-only worked example (how to add a suite).
└── *.py         # Top-level tests run alongside required/ in CI.
```

## `required/` and top-level `tests/*.py`

These run on every CI build and gate merges. They use mocks/fakes: no GPUs, no
running model servers, no kubectl. If you add a test that needs any of those,
it does **not** belong here. `tests/test_requirements.py` is one of these,
exercising the harness requirement-inference and node-matching logic with no
cluster (the harness package itself lives under `optional/`, as live-tier
infrastructure).

## `optional/` (the live tier)

These are end-to-end tests that drive real model servers (vLLM today) on a
GPU-backed Kubernetes cluster. They live outside the gating CI suite because
**CI doesn't have access to the hardware they need**: accelerator nodes, model
weights, HuggingFace tokens, the whole stack. They're not blocking; they're a
hand-run validation step before merging changes that touch wire format,
multimodal payloads, or anything else where "passes unit tests" doesn't imply
"actually works against a real server."

They are marked `@pytest.mark.live` and **auto-skipped unless you pass
`--kubeconfigs`**, so a plain `pytest tests` / `pdm run test` stays green
without anyone remembering `-m "not live"` (that filter still works too). Run
the tier explicitly, passing one or more kubeconfigs; a case is skipped when no
cluster has a live node matching its manifest's `nodeSelector`:

```sh
pytest tests/optional -m live --kubeconfigs=/path/to/kubeconfig
```

`pdm run test:e2e --kubeconfigs=/path/to/kubeconfig` is the convenient entry
point: it runs the simulated e2e suite plus this live tier (and without
`--kubeconfigs`, `pdm run test:e2e` runs just the simulated suite, since the
live cases auto-skip).

Useful flags: `--image` (or `$INFERENCE_PERF_IMAGE`) overrides the job image,
and `--sweep-orphan-namespaces` reclaims `inference-perf-e2e-*` namespaces left
by killed prior runs.

### How it works

`harness/` is shared and knows nothing about any particular suite:

- `requirements.py` reads a case's hardware requirement (its manifest
  `nodeSelector`) and the Deployment name(s) to wait on, straight from the
  manifest, and matches the requirement against live cluster nodes.
- `slots.py` is a file-based semaphore keyed by `(cluster, nodeSelector)` so
  cases contending for the same scarce hardware queue and reuse it sequentially.
- `runner.py` deploys the server, runs the `inference-perf` job, and verifies
  the request lifecycle summary. It is **not** multimodal-specific.
- `conftest.py` ties these together in the `cluster_for_case` fixture: match,
  acquire a slot, hand the test a fresh namespace, tear it down on exit.

### Adding a suite

A suite is just a folder of cases plus a one-screen test module:

1. Drop cases under `optional/<suite>/cases/<case>/`, each a `vllm.yaml` (server
   manifest, carrying the `nodeSelector` and Deployment name) and a `config.yml`
   (the inference-perf config).
2. Add `optional/test_<suite>.py` that parametrizes `cluster_for_case` by those
   manifests and calls `runner.run_case(...)`.

Nothing in `harness/` needs to change: the hardware requirement and the
Deployment to wait on are read from your manifest. `test_text.py` is a complete
worked example. If your suite needs a different run shape (a different server, a
different verification, or no `inference-perf` job at all), keep the fixture and
pass your own logic instead of `runner.run_case`.

Current suites:

- `optional/multimodal/` — multimodal benchmarking against a Qwen3-VL
  deployment; cases under `cases/*/`, driven by `test_multimodal.py`.
- `optional/text/` — a minimal text-only suite (a differently named Deployment,
  discovered from the manifest) that exists to demonstrate the pattern above.

Per-case reports are written to `cases/<case>/output/`.
