# Comparing Results Across Benchmarking Tools

Two benchmarking tools pointed at the same server with their default settings do not produce
comparable numbers. The differences are in the workload each tool generates, in what each one
counts, and in what each one excludes from the measurement window, and they move headline
throughput and latency by tens of percent before the server is involved at all. In practice a
cross-tool gap is a configuration difference until proven otherwise.

This page lists what has to agree for a comparison to mean anything, and how to check that it
did. It is the user-facing half of the parity work in #481, which encodes the same rules as
runnable workload definitions.

## What has to agree

| Dimension | Why it moves the numbers |
| :--- | :---
| Output length | Sets decode work per request, so it drives output throughput, TPOT and total tokens |
| Input length | Sets prefill work, and can push prompts across a server-side batching or context boundary |
| `ignore_eos` | Without it the model stops when it stops, and output length is a property of the model rather than of the config |
| Sampling parameters | Greedy and sampled decoding are different work, and tools differ on what they send by default |
| Load model | A fixed-concurrency (closed loop) run and a fixed-rate (open loop) run measure different things |
| Warmup | Compile, cache and autoscale effects land inside the window unless excluded |
| Token counting | Client re-tokenization and server usage disagree, and tools differ in which they report |
| Server configuration | Same server build, same flags, same model, or the comparison is of two servers |

## Output length

Pin it on both sides: a fixed output length plus `ignore_eos`, so every request generates
exactly the requested number of tokens.

In inference-perf, set `server.ignore_eos: true` and a fixed output distribution:

```yaml
server:
  ignore_eos: true
data:
  output_distribution:
    type: fixed
    mean: 1024        # every request asks for exactly this many tokens
    total_count: 1000
```

`type: fixed` emits `mean` for every request. The default is `type: normal` with a nonzero
`std_dev`, so a config that does not say otherwise is sending a distribution, not a length.

## Input length and the range-ratio footgun

`vllm bench serve` controls input length with `--random-range-ratio`, and **the meaning of that
flag was inverted upstream**, so the same value produces opposite workloads depending on the
version under test:

| vLLM version | Default (`0.0`) produces | Fixed lengths need |
| :--- | :--- | :---
| Before v0.8.4 (vllm-project/vllm#16126) | `Uniform[0, len]`, a mean of roughly half the requested length | `--random-range-ratio 1.0` |
| v0.8.4 and later | Fixed lengths | `--random-range-ratio 0.0` (the default) |

Running the older behavior against an inference-perf run with fixed lengths compares a workload
of about half the tokens against the full one. This was the root cause of a reported cross-tool
throughput gap, and it is the fixture the parity harness in #481 is built around. Version
strings printed by a serving container describe the engine, not the vendored benchmark script,
so check the behavior rather than the version when a container is involved.

The same care applies to inference-perf's `data.input_distribution`: pin it with `type: fixed`
when the goal is comparability rather than a realistic mix.

## Warmup and the measurement window

inference-perf has no dedicated warmup phase; every request it sends is measured. Tools that
exclude a warmup set are therefore reporting a different window on the same server, and the
difference is largest exactly where warmup matters most: first-token latency tails on backends
that compile or populate caches on first use.

Until a first-class warmup stage exists, approximate it with a short first stage in
`load.stages` and compare later stages only, or hold the server warm before the run and say so
alongside the numbers.

## Load model

`load.type: constant` or `poisson` offers requests at a rate, independent of how fast the
server answers. `load.type: concurrent` keeps a fixed number of requests in flight, so
throughput is a function of latency by construction. A closed-loop run and an open-loop run are
not comparable at any rate, and neither is a comparison where one tool caps per-worker
concurrency below the other's offered load.

## Which side counted the tokens

Tools differ in whether reported output tokens are the server's own count or the client's
re-tokenization of the response, and the two disagree. See
[Token Accounting and Provenance](./metrics.md#token-accounting-and-provenance) for what each
inference-perf field is derived from. For a cross-tool comparison, prefer the server-reported
count on both sides, and set `report.request_lifecycle.use_server_output_tokens: true` so the
per-token latency metrics are normalized by it.

## Server configuration

Hold the server identical: same build, same model and quantization, same
`--max-model-len`, same batching and cache flags. Note that input length interacts with this:
prompts that are longer on one side, including the tokenizer and chat-template overhead the
client does not model, can cross a chunked-prefill or context boundary and change the server's
batching, which shows up as a client-side latency difference that no client caused.

## Verify before comparing

1. Compare the total tokens each tool sent and received, not just the rates. For fixed lengths,
   totals should equal requests times length on both sides; a total that lands near half the
   expected value is the range-ratio case above.
2. Compare the input and output length statistics each tool prints. Equal means and unequal
   minima and maxima still means non-comparable workloads.
3. Check `token_count_mismatches` and `client_fallback_count` in the inference-perf report. A
   nonzero value means its own counts are mixed-source, which has to be resolved before
   attributing a delta to the other tool.
4. Only after those agree, compare throughput and latency, and report the configuration of both
   runs alongside the numbers.
