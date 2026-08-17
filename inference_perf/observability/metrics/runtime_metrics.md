# Inference-Perf Runtime Metrics

These are the Prometheus metrics inference-perf can export about its own runtime over an HTTP `/metrics` endpoint. They are distinct from the metrics inference-perf scrapes from the model server under test and from the benchmark result definitions in [metrics.md](../../../docs/metrics.md).

This document is automatically generated from the metric specs under `inference_perf/observability/metrics/sets/`. Do not edit it by hand; run `pdm run update:runtime-metrics` after changing the specs.

| Metric | Type | Labels | Exported | Description |
| --- | --- | --- | --- | --- |
| `inference_perf_run_elapsed_seconds` | Gauge | none | Always | Wall-clock seconds elapsed since the benchmark run started; 0 until the run starts. |
| `inference_perf_stages` | Gauge | none | Always | Number of load stages configured for the run. |
| `inference_perf_stage_running` | Gauge | `stage` | Always | 1 while the stage is executing, 0 once it has ended. A stage that has not started has no series. |
| `inference_perf_stage_start_timestamp_seconds` | Gauge | `stage` | Always | Unix time at which the stage started. |
| `inference_perf_stage_end_timestamp_seconds` | Gauge | `stage` | Always | Unix time at which the stage ended, whether it completed or was cut short. |
| `inference_perf_requests_in_flight` | Gauge | none | Always | Requests sent to the server and not yet finished, sampled at scrape time. |
| `inference_perf_requests_total` | Counter | `stage`, `status` | Always | Request attempts that have completed, by stage and final status. Incremented when the attempt finishes or fails, not when it is sent. |
| `inference_perf_request_errors_total` | Counter | `stage`, `error_type` | Always | Failed request attempts by stage and error class (the client's exception class or 'HTTP Error <status>'). |
| `inference_perf_prompt_tokens_total` | Counter | `stage` | Always | Prompt tokens of successful requests by stage; rate() gives input throughput. |
| `inference_perf_output_tokens_total` | Counter | `stage` | Always | Output tokens of successful requests by stage; rate() gives output throughput. Uses the server's usage.completion_tokens when reported, else the client-side count. |
| `inference_perf_request_latency_seconds` | Histogram | `stage` | Always | End-to-end latency of successful requests by stage. |
| `inference_perf_time_to_first_token_seconds` | Histogram | `stage` | Streaming runs only (api.streaming); unary responses have no token timeline. | Time to first token of successful streaming requests by stage: first content chunk minus request start. |
| `inference_perf_time_per_output_token_seconds` | Histogram | `stage` | Streaming runs only (api.streaming); unary responses have no token timeline. | Time per output token of successful streaming requests by stage: (last chunk - first chunk) / (output tokens - 1), for requests with more than one output token. |
