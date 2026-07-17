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
    }
  },
  "failures": {
    "count": 3,
    "request_latency": {
      "mean": 9.948665728999458,
      "min": 0.5831485409980814,
      "p90": 11.684405915999378
    },
    "prompt_len": {
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
- **`successes`**: Metrics for successful requests.
- **`failures`**: Metrics for failed requests, including the per-label error breakdown.
- **`goodput_metrics`**: (Optional) Goodput statistics if constraints were configured.