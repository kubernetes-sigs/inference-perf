# Benchmark Report v0.2 (BR0.2)

inference-perf can emit reports in the [BR0.2 schema](https://github.com/llm-d/llm-d-benchmark/blob/main/docs/benchmark_report.md)
defined by llm-d-benchmark. When enabled, one BR0.2 report is generated per
load-stage alongside the native lifecycle reports.

inference-perf is responsible only for the performance `results` section.
Everything else (stack configuration, run/scenario metadata) is supplied
by the user via a **partial report** — a YAML or JSON file in BR0.2 shape
with the `results` section omitted. inference-perf loads the partial,
fills in `results` from the actual run, and emits the merged report.

This is opt-in via `report.br_v0_2.partial_report`. When unset (the default),
no BR0.2 reports are produced and there is no behavior change.

## Quick start

Write a partial report describing the stack and (optionally) run metadata:

```yaml
# partial_report.yaml
run:
  eid: experiment-1
  description: "Qwen3-0.6B latency baseline"
  keywords: [baseline, latency]
scenario:
  stack:
    - metadata:
        schema_version: "0.0.1"
        label: vllm-svc-0
      standardized:
        kind: inference_engine
        tool: vllm
        tool_version: vllm/vllm:0.6.0
        role: replica
        replicas: 1
        model:
          name: Qwen/Qwen3-0.6B
        accelerator:
          model: NVIDIA-H100-80GB-HBM3
          count: 8
          parallelism:
            tp: 8
            dp: 1
            dp_local: 1
            workers: 1
      native:
        args:
          --tensor-parallel-size: "8"
        envars: {}
```

Point inference-perf at it:

```yaml
report:
  br_v0_2:
    partial_report:
      local:
        path: ./partial_report.yaml
```

A run with two stages will produce, in addition to the existing reports:

```
stage_0_benchmark_report_v0_2.yaml
stage_1_benchmark_report_v0_2.yaml
```

Each file validates against the vendored BR0.2 pydantic schema in
`inference_perf/reportgen/br/v0_2/schema.py`.

## What inference-perf fills

inference-perf is strictly responsible for the **`results`** section, derived
from request metrics observed during the run:

- `results.request_performance.aggregate.{requests, latency, throughput}`
  with the BR0.2 percentile set (`p0p1, p1, p5, p10, p25, p50, p75, p90,
  p95, p99, p99p9`) and unit annotations the schema validators require.

If the partial omits `run.uid`, inference-perf auto-generates one per stage
(required by the BR0.2 schema). Anything else (`version`, the rest of `run`,
the entire `scenario` block) is taken verbatim from the partial.

## What the partial may contain

Anything in the BR0.2 schema **except** the `results` section. Common
choices: `run.{eid, cid, pid, time, user, description, keywords}` and
`scenario.{stack, load}`. The partial is validated against the vendored
BR0.2 schema with `extra="forbid"`, so extra or mistyped fields fail the
run before any work is done.

A partial whose `results` section contains any non-null sub-field is
rejected — `results` is exclusively inference-perf's territory:

```
PartialReportError: BR0.2 partial report must not contain pre-existing
performance information; found non-empty 'results' fields: ['request_performance'].
```

An entirely empty partial is allowed; inference-perf will fill `run.uid`
and `results`, leaving everything else null.

## Sources: local or GCS

The partial can live on the local filesystem or in GCS. Exactly one
source must be set.

**Local:**

```yaml
report:
  br_v0_2:
    partial_report:
      local:
        path: ./partial_report.yaml
```

**GCS** (Application Default Credentials, same as
`report.storage.google_cloud_storage`):

```yaml
report:
  br_v0_2:
    partial_report:
      google_cloud_storage:
        bucket_name: my-benchmark-artifacts
        path: experiments/exp-1/partial.yaml
```

## Schema source

The pydantic models in `inference_perf/reportgen/br/v0_2/{base,schema_v0_2,schema_v0_2_components}.py`
are vendored from llm-d/llm-d-benchmark. The header of each file pins the
upstream commit SHA. To resync after a BR0.2 schema bump, replace those three
files from upstream and re-run `tests/reportgen/br/v0_2/test_schema_fixture.py`
to confirm round-trip validation against the upstream example still holds.
