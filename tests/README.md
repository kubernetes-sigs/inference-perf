# Tests

```
tests/
├── required/   # Run in CI on every PR. Self-contained — no external deps.
├── optional/
│   ├── automated/   # Skipped by default; opt in locally when needed.
│   └── manual/      # Run by hand against a real cluster (see below).
└── *.py        # Top-level tests run alongside required/ in CI.
```

## `required/` and top-level `tests/*.py`

These run on every CI build and gate merges. They use mocks/fakes — no GPUs, no
running model servers, no kubectl. If you add a test that needs any of those,
it does **not** belong here.

## `optional/manual/`

These are end-to-end tests that drive real model servers (vLLM today) on a
GPU-backed Kubernetes cluster. They live outside the CI suite because **CI
doesn't have access to the hardware they need** — H100 nodes, model weights,
HuggingFace tokens, the whole stack. They're not blocking; they're a
hand-run validation step before merging changes that touch wire format,
multimodal payloads, or anything else where "passes unit tests" doesn't
imply "actually works against a real server."

To run them you need a configured cluster, a built `inference-perf` image,
and the relevant secrets. They are marked `@pytest.mark.live` and excluded
from default CI via `-m "not live"`. Run them explicitly, passing one or more
kubeconfigs; a case is skipped when no cluster has a live node matching its
manifest's `nodeSelector`:

```sh
pytest tests/optional -m live --kubeconfigs=/path/to/kubeconfig
```

Useful flags: `--image` (or `$INFERENCE_PERF_IMAGE`) overrides the job image,
and `--sweep-orphan-namespaces` reclaims `inference-perf-e2e-*` namespaces left
by killed prior runs.

- `optional/multimodal/` — multimodal benchmarking against a Qwen3-VL
  deployment; cases under `cases/*/`, driven by `test_multimodal.py`. Per-case
  reports are written to `cases/<case>/output/`.

## `optional/automated/`

Skipped by default in CI; can be opted into locally for slower or
environment-dependent tests that don't need GPUs.
