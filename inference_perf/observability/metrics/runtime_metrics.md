# Inference-Perf Runtime Metrics

These are the Prometheus metrics inference-perf can export about its own runtime over an HTTP `/metrics` endpoint. They are distinct from the metrics inference-perf scrapes from the model server under test and from the benchmark result definitions in [metrics.md](../../../docs/metrics.md).

This document is automatically generated from the metric specs under `inference_perf/observability/metrics/sets/`. Do not edit it by hand; run `pdm run update:runtime-metrics` after changing the specs.

| Metric | Type | Labels | Exported | Description |
| --- | --- | --- | --- | --- |
| `inference_perf_run_elapsed_seconds` | Gauge | none | Always | Wall-clock seconds elapsed since the benchmark run started; 0 until the run starts. |
| `inference_perf_requests_total` | Counter | `stage`, `status` | Always | Request attempts that have completed, by stage and final status. Incremented when the attempt finishes or fails, not when it is sent. |
