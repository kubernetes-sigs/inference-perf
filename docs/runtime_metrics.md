# Inference-Perf Runtime Metrics

These are the Prometheus metrics inference-perf can export about its own runtime over an HTTP `/metrics` endpoint. They are distinct from the metrics inference-perf scrapes from the model server under test and from the benchmark result definitions in [metrics.md](./metrics.md).

This document is automatically generated from the metric specs under `inference_perf/observability/metrics/sets/`. Do not edit it by hand; run `pdm run update:runtime-metrics` after changing the specs.

Every metric the endpoint can expose has a row below, and nothing else can be exposed: `pdm run check:runtime-metrics` scrapes a registry built with all config gating bypassed and fails if that exposition and this table disagree in either direction. It runs inside `pdm run validate`, which is merge-blocking.

## Stability

Every metric declares a stability level, and that level is prepended to the metric's HELP text, so a scrape says what is promised without anyone having to find this file:

- `ALPHA`: May be renamed, relabeled or removed in any release, with no notice.
- `BETA`: Labels may still change, but no rename or removal without a release that deprecates it first.
- `STABLE`: Name, type and label names are fixed for the current major version.

**Every metric below is `ALPHA` today, and the whole set stays `ALPHA` through v0.7.0.** These names, labels and buckets are a first cut that we expect to refine while the endpoint gets used; nothing is promoted before v1.0.0, and promotion is per metric, one `stability=` in its spec, not a blanket graduation of the set. The level appears only in the HELP text, never in a metric name and never in a label, so promoting a metric later does not break the queries or dashboards written against it.

## Metrics

| Metric | Type | Stability | Labels | Exported | Description |
| --- | --- | --- | --- | --- | --- |
| `inference_perf_run_elapsed_seconds` | Gauge | `ALPHA` | none | Always | Wall-clock seconds elapsed since the benchmark run started; 0 until the run starts. |
| `inference_perf_stages` | Gauge | `ALPHA` | none | Always | Number of load stages configured for the run. |
| `inference_perf_stage_running` | Gauge | `ALPHA` | `stage` | Always | 1 while the stage is executing, 0 once it has ended. A stage that has not started has no series. |
| `inference_perf_stage_start_timestamp_seconds` | Gauge | `ALPHA` | `stage` | Always | Unix time at which the stage started. |
| `inference_perf_stage_end_timestamp_seconds` | Gauge | `ALPHA` | `stage` | Always | Unix time at which the stage ended, whether it completed or was cut short. |
| `inference_perf_requests_in_flight` | Gauge | `ALPHA` | none | Always | Requests sent to the server and not yet finished, sampled at scrape time. |
| `inference_perf_requests_total` | Counter | `ALPHA` | `stage`, `status` | Always | Request attempts that have completed, by stage and final status. Incremented when the attempt finishes or fails, not when it is sent. |
| `inference_perf_workers_lost_total` | Counter | `ALPHA` | `stage`, `cause` | Always | Worker processes that died during the stage, by stage and cause (the exception class the worker reported, else the signal that killed it, else its exit code). The run replaces a dead worker and carries on, so a non-zero value marks a stage that did not offer the load it was configured for: its results are not comparable with a stage that kept every worker. |
| `inference_perf_request_errors_total` | Counter | `ALPHA` | `stage`, `error_type` | Always | Failed request attempts by stage and error class (the client's exception class or 'HTTP Error <status>'). |
| `inference_perf_prompt_tokens_total` | Counter | `ALPHA` | `stage` | Always | Prompt tokens of successful requests by stage; rate() gives input throughput. |
| `inference_perf_output_tokens_total` | Counter | `ALPHA` | `stage` | Always | Output tokens of successful requests by stage; rate() gives output throughput. Uses the server's usage.completion_tokens when reported, else the client-side count. |
| `inference_perf_stage_requests_planned` | Gauge | `ALPHA` | `stage` | Always | Requests the stage set out to issue. Absent for stages whose request count is not known up front. |
| `inference_perf_stage_requests_finished` | Gauge | `ALPHA` | `stage` | Always | Requests the stage is done with, whatever became of them, including any abandoned before dispatch. Read against inference_perf_stage_requests_planned for stage completion; this is the same count the stage's own termination check reads. |
| `inference_perf_stage_requests_skipped` | Gauge | `ALPHA` | `stage` | Always | Requests counted as finished that were never sent, because the session had already failed or the request could not be built. These produce no lifecycle metric, so they appear in no outcome counter and in no report. |
| `inference_perf_stage_sessions_planned` | Gauge | `ALPHA` | `stage` | Always | Sessions the stage set out to run. Only present for session-based stages. |
| `inference_perf_stage_sessions_finished` | Gauge | `ALPHA` | `stage` | Always | Sessions the stage is done with, including any skipped because their graph could not be built. Read against inference_perf_stage_sessions_planned for stage completion. |
| `inference_perf_stage_sessions_skipped` | Gauge | `ALPHA` | `stage` | Always | Sessions counted as finished whose graph could not be built, so they dispatched nothing and produced no session lifecycle metric. |
| `inference_perf_stages_completed` | Gauge | `ALPHA` | none | Always | Stages that have ended, whether they completed or were cut short. Read against inference_perf_stages. |
| `inference_perf_request_latency_seconds` | Histogram | `ALPHA` | `stage` | Always | End-to-end latency of successful requests by stage. |
| `inference_perf_time_to_first_token_seconds` | Histogram | `ALPHA` | `stage` | Streaming runs only (api.streaming); unary responses have no token timeline. | Time to first token of successful streaming requests by stage: first content chunk minus request start. |
| `inference_perf_time_per_output_token_seconds` | Histogram | `ALPHA` | `stage` | Streaming runs only (api.streaming); unary responses have no token timeline. | Time per output token of successful streaming requests by stage: (last chunk - first chunk) / (output tokens - 1), for requests with more than one output token. |
