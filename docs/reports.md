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
  }
}
```

*(Note: Actual reports contain more percentiles and metrics).*

### Key Sections

- **`load_summary`**: Details about the requested vs achieved load.
- **`successes`**: Metrics for successful requests.
- **`failures`**: Metrics for failed requests.
- **`goodput_metrics`**: (Optional) Goodput statistics if constraints were configured.
