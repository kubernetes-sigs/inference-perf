# Benchmark Report v0.2 (BR0.2)

inference-perf can emit reports in the [BR0.2 schema](https://github.com/llm-d/llm-d-benchmark/blob/main/docs/benchmark_report.md)
defined by llm-d-benchmark. When enabled, one BR0.2 report is generated per
load-stage alongside the native lifecycle reports.

This is opt-in via `report.br_v0_2.enabled`. When disabled (the default), no
BR0.2 reports are produced and there is no behavior change.

## Quick start

```yaml
report:
  br_v0_2:
    enabled: true
    description: "Qwen3-0.6B latency baseline"
    keywords: [baseline, latency]
```

A run with two stages will produce, in addition to the existing reports:

```
stage_0_benchmark_report_v0_2.yaml
stage_1_benchmark_report_v0_2.yaml
```

Each file validates against the vendored BR0.2 pydantic schema in
`inference_perf/reportgen/br/v0_2/schema.py`.

## What gets populated

inference-perf produces the **workload-side** sections of BR0.2 entirely from
its own state:

- `version`, `run.{uid,time,user,pid,description,keywords}`
- `scenario.load.{metadata,standardized,native}` — distributions, prefix,
  multi-turn, rate/concurrency, full inference-perf config under `native.config`
- `results.request_performance.aggregate.{requests,latency,throughput}` with
  the BR0.2 percentile set (`p0p1, p1, p5, p10, p25, p50, p75, p90, p95, p99,
  p99p9`) and unit annotations the schema validators require

The **deployment-side** sections (`scenario.stack[]`, `results.observability`,
`results.component_health`) come from a `StackObservabilityCollector`. There
are three:

| Collector | When used | Produces |
|---|---|---|
| `NoopStackCollector` | BR0.2 disabled | nothing (empty stack/observability) |
| `ConfigStackCollector` | `report.br_v0_2.stack` is set, OR no K8s available | passes through user-supplied `scenario.stack[]` overrides |
| `KubernetesStackCollector` | In-cluster, or `report.br_v0_2.kubernetes` is set, AND the `k8s` extra is installed | live K8s discovery (see below) |

Auto-selection: a manually-supplied `stack` always wins (the migration path
for llm-d-benchmark to drop its translator). Otherwise K8s is used when
available, falling back to a `ConfigStackCollector` with an empty stack.

## Manual override (running outside K8s)

Provide stack components directly. Each entry mirrors the BR0.2
`Component` shape and is validated by the vendored pydantic schema:

```yaml
report:
  br_v0_2:
    enabled: true
    stack:
      - metadata:
          schema_version: "0.0.1"
          label: vllm-svc-0
          cfg_id: <hash>
        standardized:
          kind: inference_engine
          tool: vllm
          tool_version: vllm/vllm:0.6.0
          role: replica            # one of: prefill, decode, replica
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

## In-cluster discovery

Install with the k8s extra:

```bash
pip install 'inference-perf[k8s]'
```

Then enable K8s discovery:

```yaml
report:
  br_v0_2:
    enabled: true
    kubernetes:
      namespace: my-namespace          # optional; auto-detected from service account
      label_selectors:                  # narrow which pods are part of the stack
        app.kubernetes.io/part-of: llm-d
      scrape_interval_seconds: 15
```

What gets collected:

- **`scenario.stack[]`** — `ComponentInspector` lists pods matching the label
  selectors, derives `tool`/`tool_version` from container image, lifts CLI
  args + env into `native`, parses `--tensor-parallel-size` etc into
  `accelerator.parallelism`, and reads `nvidia.com/gpu.product` from node
  labels for `accelerator.model`. Role comes from `llm-d.ai/role`,
  `inference.networking.x-k8s.io/role`, or `app.kubernetes.io/component`.
- **`observability.pod_startup_times`** — `PodLifecycleWatcher` watches pods
  in the namespace and records the time from `creationTimestamp` to the
  `Ready=True` condition.
- **`observability.replica_status`** — `ReplicaStatusSampler` polls
  Deployments + StatefulSets at the configured interval and snapshots
  desired/available/ready/updated counts.
- **vLLM/EPP per-pod metrics** — `PerPodPromScraper` GETs each discovered
  pod's `/metrics` endpoint (port from the `prometheus.io/port` annotation,
  defaulting to 8000), parses the Prometheus text format, and aggregates
  per-pod statistics for known metrics
  (`vllm:gpu_cache_usage_perc`, `vllm:num_requests_running`,
  `vllm:num_requests_waiting`, `vllm:num_preemptions_total`,
  `vllm:prefix_cache_*`, `inference_extension:inference_pool:*`).

The watcher and sampler run in the background between `start()` (before
stages execute) and `stop()` (after stages complete).

## Required RBAC

When the `KubernetesStackCollector` is in use, the inference-perf pod's
ServiceAccount needs:

```yaml
- apiGroups: [""]
  resources: [pods, nodes]
  verbs: [get, list, watch]
- apiGroups: [apps]
  resources: [deployments, statefulsets]
  verbs: [get, list]
```

## Schema source

The pydantic models in `inference_perf/reportgen/br/v0_2/{base,schema_v0_2,schema_v0_2_components}.py`
are vendored from llm-d/llm-d-benchmark. The header of each file pins the
upstream commit SHA. To resync after a BR0.2 schema bump, replace those three
files from upstream and re-run `tests/reportgen/br/v0_2/test_schema_fixture.py`
to confirm round-trip validation against the upstream example still holds.
