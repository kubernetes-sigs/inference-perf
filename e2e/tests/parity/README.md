# Tool-parity cases

## The problem this solves

inference-perf and `vllm bench serve` do the same job: send a stream of
requests at an LLM server and time the answers. When the two print different
numbers for "the same" benchmark, the first question is whether they were
actually asked to send the same traffic. In the past the answer has often been
no. One tool's `range_ratio: 0` meant "random lengths" while the other's meant
"fixed". One tool sends a warmup request the other does not. One spaces
requests evenly, the other spaces them randomly. Each of those took days to
find because nothing checked it. This directory is that check.

Words used below:

- **workload** or **offered load**: the traffic a tool sends. How many
  requests, how long each prompt is, how many output tokens it asks for, how
  fast it sends them, and how many it keeps in flight at once.
- **case**: one workload written down twice, once per tool, plus the numbers
  both versions must produce.
- **absorber**: the fake server both tools are pointed at (next section).

## How it works

Both tools are pointed at the **absorber** (`e2e/utils/absorber.py`). It does
not run a model. It answers on the usual OpenAI-style URLs, writes down every
request it receives (when it arrived, when its reply finished, the full
request body), and replies with filler text streamed at a fixed pace so the
tools behave as they would against a real server.

After each tool runs, the absorber's list of recorded requests is the truth
about what that tool sent. The test checks that list against the numbers in
`expected.yaml`, then checks the two tools' lists against each other. A
failure names the setting that leaked: request count, prompt lengths,
`max_tokens`, the `stream` or `ignore_eos` flags, arrival rate, or
concurrency.

Rate and concurrency are not computed here. The arrival/finish times go
through `e2e/utils/load_shape.py`, the same helpers the load-shape accuracy
test (`e2e/tests/test_load_shape_accuracy_sim.py`, #633) uses to check
inference-perf against its own config: the same sweep line for "how many were
in flight", and the same `rate_tolerance(n, arrival)` for "how far off the
configured rate is still fine". One definition of each, shared. The only
difference is where the timestamps come from: that test reads the client's
own, this one reads the absorber's.

Only the *sent* traffic is compared. The latency and throughput numbers each
tool prints are never looked at here. Those are the tools' output; this test
is about their input.

## What a case looks like

Every subdirectory of `cases/` is one case. The test finds them by listing the
directory, so adding a case means adding three files and no Python:

```
cases/<name>/
  inference-perf.yaml   # a normal inference-perf config for the workload
  vllm-bench.args       # `vllm bench serve` flags, one per line, # comments ok
  expected.yaml         # the numbers both tools must hit, plus tolerances
```

The test fills in the parts that depend on where the absorber happens to be
running; everything else in the two files is used exactly as written:

- `inference-perf.yaml`: `server.base_url` and
  `tokenizer.pretrained_model_name_or_path` are replaced. The bundled
  gemma-3-270m tokenizer is used, so cases run without network access.
- `vllm-bench.args`: `--base-url`, `--model`, `--tokenizer`, `--save-result`
  and `--result-filename` are added to the end.

One thing to know: `tests/required/config/test_yaml_configs.py` schema-checks
every YAML file under `e2e/`. That is good for `inference-perf.yaml` (a case
breaks loudly if the config format changes) and wrong for `expected.yaml`
(it is not an inference-perf config). `expected.yaml` files under `cases/` are
skipped there by a wildcard, so new cases still need no Python edit.

## expected.yaml

```yaml
absorber:                 # how fast the absorber replies
  ttft_ms: 40             # wait before the first chunk
  itl_ms: 5               # wait between later chunks
workload:
  num_requests: 600       # exact, per tool, after dropping declared warmup requests
  prompt_tokens: 128      # per request, measured by re-tokenizing the prompt text
  prompt_tokens_rel_tol: 0.15
  max_tokens: 64          # exact, every request
  stream: true            # exact
  ignore_eos: true        # exact
load:
  mode: rate              # "rate" (open loop) or "concurrency" (closed loop)
  rate: 40.0              # rate mode: average requests per second actually sent
  # concurrency: 8        # concurrency mode: never above 8 in flight, and on
                          # average within half a slot of 8 in steady state
tools:
  inference-perf:
    arrival: constant     # rate mode: how this tool spaces requests (sets its tolerance)
  vllm-bench:
    arrival: poisson
    leading_extra_requests: 1   # vllm bench sends one warmup request first
```

`leading_extra_requests` says how many of a tool's earliest requests are
warmup traffic to ignore before checking anything. If a tool needs a nonzero
value here, that is itself a difference between the tools, and it stays
written down in the case file on purpose.

`arrival` is required in rate mode and is either `constant` (evenly spaced)
or `poisson` (random gaps). There is no rate tolerance number in the file:
the allowed wobble is `rate_tolerance(num_requests, arrival)` from
`load_shape.py`, which at 600 requests is 5% for `constant` and about 16% for
`poisson` (four standard deviations of the random spacing). The way to
tighten a case is to raise `num_requests`, never to edit a number.

## Differences that are expected and allowed for

- **Spacing between requests.** At the same rate, `vllm bench serve` picks
  random gaps between requests (Poisson) while inference-perf's `constant`
  load spaces them evenly. Cases check the average rate only, never the
  spacing pattern, and each tool gets the tolerance its own spacing earns
  (`tools.<tool>.arrival`). When the two tools are compared directly, the
  looser of the two tolerances applies, since the evenly spaced tool adds
  almost no wobble of its own.
- **Prompt length is approximate.** Both tools aim for `prompt_tokens` under
  the same tokenizer, but text built from random token ids does not always
  tokenize back to the same count. Hence `prompt_tokens_rel_tol` instead of an
  exact match.
- **vllm flags change meaning between versions.** The args files are written
  for the vllm version pinned in `e2e/utils/vllm_bench.py` (`VLLM_PINNED_REF`),
  and every flag was checked against that version's
  `vllm/benchmarks/serve.py` and `vllm/benchmarks/datasets.py`. In particular
  `--random-range-ratio 0` means "fixed lengths" at this pin, but it has meant
  "anywhere from 0 to the full length" in other versions. That mix-up is what
  motivated #481. Re-check both files when bumping the pin.
- **Some things cannot be said on both sides.** The pinned vllm always sends
  `stream: true` on the completions endpoint, so a `stream: false` case cannot
  be written for the vllm leg today. vllm also subtracts the tokenizer's
  special-token count from `--random-input-len` before sampling, so a tokenizer
  that adds a BOS token ends up sending `len-1` prompt tokens. That is another
  reason `prompt_tokens_rel_tol` exists.

## Running

```
pdm run pytest e2e/tests/parity
```

The inference-perf side always runs; the absorber needs nothing installed. The
vllm side needs a runnable vllm: either set `$VLLM_BENCH_BIN` to one you
already have, or set `$VLLM_BENCH_PROVISION=1` to let the test clone and
install the pinned version once (large download, see
`e2e/utils/vllm_bench.py`). Without either, the vllm-side tests skip.

The vllm side has not been run yet. The args files and
`leading_extra_requests` were worked out by reading the pinned vllm source, not
by running it. Treat the first real vllm run as the thing that checks this
directory, not as a regression check.
