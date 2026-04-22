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
and the relevant secrets. See each subdirectory for the harness scripts and
per-case configs:

- `optional/manual/multimodal/` — multimodal benchmarking against a Qwen3-VL
  deployment. Driven by `e2e_test.sh`; cases under `cases/*/`.

## `optional/automated/`

Skipped by default in CI; can be opted into locally for slower or
environment-dependent tests that don't need GPUs.
