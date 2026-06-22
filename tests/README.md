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
cluster has a live node satisfying its manifest's GPU `nodeAffinity`:

```sh
pytest tests/optional -m live --kubeconfigs=/path/to/kubeconfig
```

**Cross-provider GPU targeting.** A case pins its GPU SKU with a required
`nodeAffinity` whose `nodeSelectorTerms` are OR'd: one term for GFD's
`nvidia.com/gpu.product` (any cluster running the NVIDIA GPU Operator, e.g. EKS/AKS)
and one for GKE's managed `cloud.google.com/gke-accelerator`. The scheduler honors
whichever the cluster uses, so the same manifest is portable across GKE/EKS/AKS with
no per-cloud copy and no setup on your existing GKE cluster. To support another
provider's label, add a term to the manifests.

`pdm run test:e2e --kubeconfigs=/path/to/kubeconfig` is the convenient entry
point: it runs the simulated e2e suite plus this live tier (and without
`--kubeconfigs`, `pdm run test:e2e` runs just the simulated suite, since the
live cases auto-skip).

Useful flags: `--image` (or `$INFERENCE_PERF_IMAGE`) overrides the job image,
and `--sweep-orphan-namespaces` reclaims `inference-perf-e2e-*` namespaces left
by killed prior runs.

### How it works

`harness/` is shared and knows nothing about any particular suite:

- `requirements.py` reads a case's hardware requirement (its manifest's required
  `nodeAffinity`) and the Deployment name(s) to wait on, straight from the
  manifest, and matches the requirement against live cluster nodes. It only reads
  the affinity; the manifest is applied as-is, so the scheduler does the enforcing.
- `slots.py` is a file-based semaphore keyed by `(cluster, requirement)` so
  cases contending for the same scarce hardware queue and reuse it sequentially.
- `runner.py` deploys the server, runs the `inference-perf` job, and verifies
  the request lifecycle summary. It is **not** multimodal-specific.
- `conftest.py` ties these together in the `cluster_for_case` fixture: match,
  acquire a slot, hand the test a fresh namespace, tear it down on exit.

### Adding a suite

A suite is just a folder of cases, no Python required:

1. Drop cases under `optional/<suite>/cases/<case>/`, each a `vllm.yaml` (server
   manifest, carrying the GPU `nodeAffinity` and Deployment name) and a `config.yml`
   (the inference-perf config).

That's it. `test_live.py` discovers every `<suite>/cases/<case>/vllm.yaml` by
glob and runs it; the case id is `<suite>/<case>`, so `-k <suite>` selects one
suite. Nothing in `harness/` needs to change: the hardware requirement and the
Deployment to wait on are read from your manifest. If your suite needs a
different run shape (a different server, a different verification, or no
`inference-perf` job at all), add a small test module that parametrizes
`cluster_for_case` and passes your own logic instead of `runner.run_case`.

Current suites:

- `optional/multimodal/` — multimodal benchmarking against a Qwen3-VL
  deployment; cases under `cases/*/`.
- `optional/text/` — a minimal text-only suite (a differently named Deployment,
  discovered from the manifest) that exists to demonstrate the pattern above.

Both run via `optional/test_live.py`; there is no per-suite test module.

Per-case reports are written to `cases/<case>/output/`.
