# Tool-parity cases

Each subdirectory of `cases/` is one parity case: a workload described twice,
once per tool, plus the invariants both descriptions must satisfy. The test
(`test_offered_load_parity.py`) discovers every case directory automatically,
so adding a case means adding files, not editing Python.

A case that holds proves the two configs describe the same workload. A case
that fails names the knob that leaked: request count, prompt lengths,
`max_tokens`, sampling flags, arrival rate, or concurrency.

## How it works

Both tools are pointed at an **absorber** (`e2e/utils/absorber.py`): an
OpenAI-compatible server whose only job is to record every request it
receives, while streaming realistically paced responses so closed-loop tools
behave as they would against a real server. The recorded requests are the
oracle. Nothing is asserted about either tool's *reported metrics* here; what
is compared is the load each tool actually *offered*. Historically that is
where the discrepancies live (range-ratio semantics, warmup requests, arrival
process, worker caps), not in the runners.

## Case layout

```
cases/<name>/
  inference-perf.yaml   # complete inference-perf config for the workload
  vllm-bench.args       # `vllm bench serve` args, one per line, # comments ok
  expected.yaml         # the invariants + absorber pacing + tolerances
```

The harness owns endpoint wiring and overwrites/appends it:

- `inference-perf.yaml`: `server.base_url` and
  `tokenizer.pretrained_model_name_or_path` are replaced (the bundled
  gemma-3-270m tokenizer is used, so cases run offline).
- `vllm-bench.args`: `--base-url`, `--model`, `--tokenizer`, `--save-result`,
  `--result-filename` are appended. Everything workload-shaped stays in the
  file, verbatim.

Adding a case needs no Python edit anywhere. The one non-obvious coupling is
that `tests/required/config/test_yaml_configs.py` schema-validates every YAML
under `e2e/`: `inference-perf.yaml` is validated there (which is what keeps
these cases from rotting as the config schema moves), and
`cases/*/expected.yaml` is excluded by a glob in that file's
`NON_CONFIG_GLOBS`, so new case directories are covered without touching it.

## expected.yaml

```yaml
absorber:                 # response pacing the absorber applies
  ttft_ms: 40
  itl_ms: 5
workload:
  num_requests: 40        # exact, per tool, after trimming leading extras
  prompt_tokens: 128      # per request, re-encoded with the shared tokenizer
  prompt_tokens_rel_tol: 0.15
  max_tokens: 64          # exact, every request
  stream: true            # exact
  ignore_eos: true        # exact
load:
  mode: rate              # "rate" | "concurrency"
  rate: 8.0               # rate mode: realized mean arrival rate
  rate_rel_tol: 0.25
  # concurrency: 8        # concurrency mode: exact max in-flight
tools:
  vllm-bench:
    leading_extra_requests: 1   # vllm bench sends one initial test request
```

`leading_extra_requests` documents per-tool warmup traffic: that many
earliest-arriving requests are excluded before asserting. That a tool needs a
nonzero value here is itself a parity finding worth keeping visible.

## Known, deliberate asymmetries

- **Arrival process**: at the same request rate, `vllm bench serve` draws
  Poisson inter-arrivals while inference-perf's `constant` load is evenly
  spaced. Cases assert the realized *mean* rate, never the shape.
- **Prompt length recovery**: both tools target `prompt_tokens` under the same
  tokenizer, but text generated from sampled token ids does not always
  re-encode to the same count, hence `prompt_tokens_rel_tol` instead of
  exactness.
- **vllm flag semantics are version-dependent.** The args files target the
  vllm pinned in `e2e/utils/vllm_bench.py` (`VLLM_PINNED_REF`), and every flag
  in them was read off that tag's `vllm/benchmarks/serve.py` and
  `vllm/benchmarks/datasets.py`. Notably `--random-range-ratio` changed meaning
  across vllm versions: under the pinned CLI the sampled range is
  `[len*(1-r), len*(1+r)]` with an inclusive upper bound, so `0` means fixed
  lengths. This is the exact footgun that motivated #481. Re-read both files
  when bumping the pin.
- **Not everything is expressible on both sides.** The pinned vllm hardcodes
  `stream: true` on the completions endpoint, so a `stream: false` case cannot
  be written for the vllm leg today. `--random-input-len` is also reduced by
  the tokenizer's special-token count before sampling, so a BOS-prepending
  tokenizer offers `len-1` prompt tokens, which is one of the reasons
  `prompt_tokens_rel_tol` exists.

## Running

The vllm leg needs a runnable vllm: set `$VLLM_BENCH_BIN`, or opt into a heavy
one-time provision with `$VLLM_BENCH_PROVISION=1` (see `e2e/utils/vllm_bench.py`).
Without one, vllm-side tests skip and the inference-perf side still asserts
against `expected.yaml`. The absorber itself needs no external binary.

The vllm leg has never been executed: the args and `leading_extra_requests`
were derived by reading the pinned source, not by running it. Treat a first
vllm run as the thing that validates this file, not as a regression check.

```
pdm run pytest e2e/tests/parity
```
