# Troubleshooting: Interpreting Anomalous Results

The other docs point in the forward direction: [config.md](./config.md) explains how to configure a run, [loadgen.md](./loadgen.md) how load is generated, [analysis.md](./analysis.md) how to chart the results. This guide points in the reverse direction: the run completed successfully, but the output looks wrong. An ITL lower than the hardware could possibly produce, throughput above the accelerator's ceiling, a latency-throughput curve that bends backwards, an empty server-metrics field.

A suspicious number is not yet a wrong number. Every anomaly falls into one of three root-cause buckets, and the job of this guide is to route you to the right one cheaply:

1. **The measurement is wrong.** The tool mismeasured or misreported. The canonical case is [#564](https://github.com/kubernetes-sigs/inference-perf/issues/564), where per-chunk re-tokenization inflated `output_len` and deflated ITL for certain tokenizers, and ran silent for about two months because nothing checked plausibility. Client-side vs server-side token count disagreement ([#580](https://github.com/kubernetes-sigs/inference-perf/issues/580), [#619](https://github.com/kubernetes-sigs/inference-perf/issues/619)) lives here too.
2. **The workload is not what you think you asked for.** The tool measured faithfully, but the run you configured is not the run you meant: a missing `ignore_eos`, a requested rate the server cannot attain so the run silently measures saturation instead, stages too short for stable tail percentiles, generation-length differences confounding a throughput comparison ([#481](https://github.com/kubernetes-sigs/inference-perf/issues/481)).
3. **The system really behaves that way.** Batching knees, queueing blowup past saturation, prefix caching making repeated-prompt runs look impossibly fast, preemption producing non-monotonic curves. Real behavior, worth understanding, nothing to fix in the benchmark.

There is also a fourth outcome that is not a root cause: **the result is valid but not comparable.** A run can pass every check below and still be inadmissible to a particular comparison (a shared dashboard, a regression baseline, a published sweep) because the comparison imposes constraints beyond validity. That case is covered [at the end](#valid-but-not-comparable), because the remedy is different: the numbers are not wrong, they just cannot sit next to those other numbers.

The buckets are ordered by how cheaply they can be ruled out. Bucket 1 is arithmetic you can do on the report in minutes. Bucket 2 is a config audit. Bucket 3 is the residual: you are only entitled to the conclusion "the system really behaves that way" after buckets 1 and 2 come up clean.

## The Diagnosis Procedure

Work through the steps in order. Each symptom entry in the [symptom index](#symptom-index) points into this procedure with its specific checks.

### Step 0: Check the automated findings

Automated report validation ([#705](https://github.com/kubernetes-sigs/inference-perf/pull/705), in review) emits a `validation.json` alongside the other report files, containing findings at two severities:

- An **error** means the report set is internally inconsistent: the tool's own math does not reconcile. This is bucket 1 by definition, and it is terminal. The run is invalid, and the remedy is to file an inference-perf bug with `validation.json` attached. Do not spend effort interpreting the other numbers.
- A **warning** means something is suspicious but not necessarily wrong. Warnings are entry points into the checks below, not verdicts.
- A **clean** report means the mechanical consistency checks in Step 1 have already been done for you, and you can move to Step 2.

Until `validation.json` is available in your build, run the Step 1 checks by hand. They are the same invariants: the automated checks and this guide are the machine-executed and human-executed halves of one invariant set. (The strict forms also run in CI against a simulated and a real vLLM server; see the helpers in `e2e/utils/accuracy.py`.)

Note the deliberate scope of "error" here: it flags a bug in inference-perf, never a misbehaving model server. A run with many failed requests is a successful benchmark of an unhealthy server and validates cleanly. But before comparing latency statistics from such a run, check `failures.count` and the `by_label` breakdown in the summary report: success-only aggregates from a run with heavy failures describe the survivors, not the offered workload.

### Step 1: Plausibility arithmetic (rules bucket 1 in or out)

All of these use fields already in the reports (see [reports.md](./reports.md)) and the metric definitions in [metrics.md](./metrics.md). No rerun needed.

**Token accounting.** For a sample of entries in `per_request_lifecycle_metrics.json`, compare the client-derived count against the server-reported count:

- client: `info.response_metrics.output_tokens` (re-tokenization of the received text)
- server: `info.response_metrics.server_usage.completion_tokens` (the server's own `usage` block, when the server sends one)

These should agree within a few tokens. (Known caveat: models that emit reasoning tokens on a separate channel can legitimately diverge here; see [#619](https://github.com/kubernetes-sigs/inference-perf/issues/619).) A large or systematic gap means the token counts feeding every derived metric (ITL, TPOT, token throughput) are wrong, and the run is a bucket 1 case. This exact divergence was how [#564](https://github.com/kubernetes-sigs/inference-perf/issues/564) was ultimately caught.

**Latency identities.** Per request, end-to-end latency decomposes as `e2e ≈ TTFT + ITL × (output_tokens − 1)` (see [metrics.md](./metrics.md) for the exact definitions). If the reported aggregates cannot be reconciled with this identity even approximately, some component was mismeasured.

**Recompute a throughput from raw data.** Output token throughput is `total output tokens / duration`. Sum `output_tokens` over the per-request entries of one stage and divide by the stage duration; it should match the stage report's throughput block. A mismatch means aggregation is broken, not the server.

**Physical floors and ceilings.** Two back-of-envelope bounds catch most impossible numbers:

- *Per-request ITL floor.* Decode is typically memory-bandwidth-bound: each token requires streaming the model weights once, so a single sequence cannot decode faster than roughly `weights_bytes / memory_bandwidth` per token. A 16 GB model on a 3.3 TB/s accelerator has an ITL floor around 5 ms. Reported mean ITL well below the floor is bucket 1 until proven otherwise (the "proven otherwise" cases are real: speculative decoding and multi-token prediction legitimately break this bound; quantization lowers `weights_bytes`).
- *Aggregate throughput ceiling.* Generating one token costs roughly `2 × parameter_count` FLOPs per sequence, so aggregate output throughput cannot exceed roughly `accelerator_FLOPs / (2 × parameter_count)` even at perfect utilization. Throughput above this is miscounted tokens or a mismeasured duration.

### Step 2: Workload audit (rules bucket 2 in or out)

The question here is whether the run you got is the run you asked for.

**Requested vs achieved rate.** Every stage report carries a `load_summary` with `requested_rate` and `achieved_rate`. If `achieved_rate` falls short of `requested_rate`, the stage did not measure the load you configured: it measured the server's saturation point, and every latency number in it is a saturation number. This is the single most common source of "weird" high-rate data points. See [loadgen.md](./loadgen.md) for how request scheduling behaves when the server cannot keep up.

**Output length control.** If the comparison depends on throughput or per-token latencies, check whether `ignore_eos: true` was set (see [config.md](./config.md)). Without it, the model chooses its own output lengths, and two runs (or two stages, or two servers) can differ in mean generation length; a throughput difference between them then confounds "faster system" with "shorter answers". Compare the output token distributions of the things being compared before believing a throughput delta.

**Stage duration vs tail percentiles.** A p99 needs on the order of hundreds of completed requests to be stable. At low request rates, short stages produce noisy tails, and noisy tails produce non-monotonic curves. Check requests-per-stage before interpreting tail percentile movement across stages.

**Dataset and cache effects.** If the dataset repeats prompts (or shares long prefixes), a server with prefix caching serves a partly-cached workload. That is a legitimate thing to measure on purpose and a confound when measured by accident.

### Step 3: Known mechanisms (bucket 3, the residual)

If Steps 0 through 2 are clean, the anomaly is probably real. The common mechanisms, with their signatures:

- **Batching knee.** Latency is flat while the batch has headroom, then climbs steeply once the server saturates. This is the knee [analysis.md](./analysis.md) tells you to look for; it is the expected shape, not an anomaly.
- **Queueing past saturation.** Beyond the knee, offered load above capacity turns into queue time. E2e latency grows without bound while throughput stays flat at capacity. Points past saturation describe queue depth, not the server.
- **Continuous batching under load.** TTFT can stay flat while ITL degrades: admission into the running batch stays fast while each decode step slows as the batch grows. Flat TTFT with climbing e2e is the classic signature.
- **Preemption and cache thrash.** Near capacity, servers may preempt sequences or evict KV cache, producing genuinely non-monotonic latency curves that reproduce across reruns.
- **Speculative decoding / multi-token prediction.** Per-token latencies below the single-forward-pass floor, legitimately.

The distinguishing property of bucket 3 is reproducibility under variation: the effect survives a rerun, a longer stage, a different seed. Bucket 1 and 2 artifacts usually do not.

## Symptom Index

| Symptom | First check (bucket 1) | Then (bucket 2) | If clean (bucket 3) |
| :--- | :--- | :--- | :--- |
| ITL lower than physically plausible | Token accounting: client vs `server_usage.completion_tokens`. Inflated client counts deflate ITL (the [#564](https://github.com/kubernetes-sigs/inference-perf/issues/564) mechanism). ITL floor arithmetic (Step 1). | Chunks carrying multiple tokens change what "inter-token" means; check `chunk_times` count vs token count for a sample request. | Speculative decoding or multi-token prediction on the server. |
| Output throughput exceeds hardware ceiling | Token accounting, then recompute throughput from per-request data; check the stage duration used as the denominator. | Very short stages measuring a warm burst; tokenizer mismatch between client and server. | Your ceiling estimate is wrong (quantization, more accelerators behind the endpoint than assumed). |
| Latency-throughput curve is non-monotonic | Confirm stage files in the analyzed directory all belong to one run (analysis reads every `stage_*_lifecycle_metrics.json` in the directory). | `requested_rate` vs `achieved_rate` per stage (points past saturation cluster and fold back); requests-per-stage too low for stable tails; output length drift across stages. | Preemption or KV-cache thrash near capacity; reproduces on rerun. |
| TTFT flat but e2e keeps climbing | Decompose per-request: is it ITL growing or `output_tokens` growing? If neither reconciles with e2e, it is a measurement bug. | Output lengths growing with load (no `ignore_eos`); truncation differences across stages. | Continuous batching: fast admission, slower decode steps as batch grows. Expected under rising load. |
| Server-metrics field empty in the report | Not a metrics bug in the usual sense: the collector could not reach or parse the server's metrics. Check the run logs for scrape errors. | Metrics collection unconfigured or pointed at the wrong endpoint (see [config.md](./config.md#metrics-collection)); metric names differ across server versions. Confirm by curling the server's `/metrics` endpoint directly. | The server version genuinely does not export that metric. |

## What To Do With the Verdict

- **Bucket 1 (measurement wrong):** the run is invalid. File an inference-perf bug with the report directory (and `validation.json` once available) attached. Do not interpret or publish the numbers; do not "correct" them by hand.
- **Bucket 2 (workload wrong):** the run is a valid measurement of a workload you did not intend. Fix the config and rerun. Keep the old reports if the accidental measurement is itself informative (saturation behavior often is).
- **Bucket 3 (real behavior):** keep the result and annotate the mechanism. If it changes the conclusion of a comparison, that is the comparison working as intended.

## Valid But Not Comparable

A comparison set imposes constraints beyond validity, because comparability is always relative to a spec: which dataset and seed, whether `ignore_eos` is set, the sweep shape (a single axis swept upward monotonically), stage durations long enough for stable tails and equal across runs, hardware metadata recorded so the runs can be grouped at all.

A run that fails one of these is not wrong. Its numbers stand as a measurement of what it measured. It just cannot sit in that comparison, and downstream consumers that aggregate benchmark reports across runs (dashboards, regression gates) may exclude or flag it. The remedy is to rerun under the comparison's constraints, not to adjust the data.

If you are producing runs for such a consumer, check its constraints before the run, not after: every constraint above is a config decision, and all of them are cheaper to set than to discover in a chart.
