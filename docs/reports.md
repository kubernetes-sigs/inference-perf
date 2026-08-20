# Inference Perf Reports

`inference-perf` generates detailed reports in JSON format after a benchmark run. These reports help you analyze the performance in depth.

## Report Files

By default, reports are saved in a directory named `reports-YYYYMMDD-HHMMSS/`. The following files are typically generated:

- **`summary_lifecycle_metrics.json`**: Aggregated metrics for the entire benchmark run.
- **`stage_N_lifecycle_metrics.json`**: Metrics for a specific load stage (where N is the stage index).
- **`per_request_lifecycle_metrics.json`**: Raw data for every single request, including timestamps and token counts.
- **`config.yaml`**: A copy of the configuration used for the run.

## Understanding the Report Structure

Here is an example snippet from a `summary_lifecycle_metrics.json` report:

```json
{
  "successes": {
    "count": 480,
    "latency": {
      "request_latency": {
        "mean": 3.31,
        "median": 2.11,
        "p90": 5.94
      },
      "time_to_first_token": {
        "mean": 0.80,
        "median": 0.20,
        "p90": 2.26
      }
    },
    "throughput": {
      "requests_per_sec": 1.02,
      "total_tokens_per_sec": 676.12
    },
    "finish_reasons": {
      "length": 478,
      "stop": 2
    },
    "output_shortfalls": 2
  },
  "failures": {
    "count": 3,
    "request_latency": {
      "mean": 9.948665728999458,
      "min": 0.5831485409980814,
      "p90": 11.684405915999378
    },
    "prompt_tokens": {
      "total": 0.0,
      "cached": 0.0,
      "uncached": 0.0,
      "mean": 0.0,
      "min": 0.0,
      "p90": 0.0,
    },
    "by_label": {
      "504 - Gateway Timeout": {
        "count": 2,
        "messages": [
          {
            "message": "...504 Gateway Time-out...",
            "session_ids": [
              "trace1715_066de3655406_a9687407",
              "trace2210_1f9b0c4d7e21_b3c58120"
            ]
          }
        ]
      },
      "400 - Invalid JSON": {
        "count": 1,
        "messages": [
          {
            "message": "...Invalid JSON: EOF while parsing a string at line 202 column 31...",
            "session_ids": [
              "trace42_9f000393d262_f395c930"
            ]
          }
        ]
      }
    }
  }
}
```

*(Note: Actual reports contain more percentiles and metrics).*

### Key Sections

- **`load_summary`**: Details about the requested vs achieved load.
- **`successes`**: Metrics for successful requests. Two fields describe whether those requests ran to the length they asked for:
  - `finish_reasons`: how many successful requests ended with each reason the server reported, verbatim (OpenAI `finish_reason`: `stop`, `length`, `tool_calls`, ...; Anthropic `stop_reason`: `end_turn`, `max_tokens`, ...). `length` and `max_tokens` mean the requested budget was delivered; anything else means the server halted on its own. Requests whose server reported no reason are not counted.
  - `output_shortfalls`: how many successful requests delivered fewer output tokens than their `max_tokens` asked for. Delivered means the server's own `usage.completion_tokens` when it reported one, otherwise the client-side count. Without `ignore_eos` a shortfall is usually the model emitting EOS as intended, which is why it is an observation here rather than a failure.
- **`failures`**: Metrics for failed requests, including the per-label error breakdown. A request fails on a non-200 status, on a transport error, or on a 200 whose body is not a completion: a body carrying a top-level `error` object (label `inbanderror`, `error_msg` is that object) or one with neither completion content nor `usage` (label `emptyresponseerror`). With `server.ignore_eos: true` (the default) a completion that delivered fewer output tokens than its `max_tokens` is also a failure (label `truncatedresponseerror`, `error_msg` names delivered-of-requested and the `finish_reason`): the server was asked to generate the full length and did not, whether it stopped early or capped the request. A server that ignores the `ignore_eos` field will report every natural stop this way; set `ignore_eos: false` for it. The body is kept as `response` in the per-request report in every case, and each per-request entry records the request's `max_tokens` alongside `info.response_metrics.finish_reason`.
- **`goodput_metrics`**: (Optional) Goodput statistics if constraints were configured.