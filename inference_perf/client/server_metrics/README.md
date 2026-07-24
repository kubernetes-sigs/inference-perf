# Model Server Metrics Query Clients

This repository provides clients to query performance metrics from various monitoring platforms. Each model server exposes a list of relevant performance metrics, and these clients are designed to retrieve and process that data effectively.

These clients consume metrics from the model server's monitoring stack. For the Prometheus metrics inference-perf itself exports about its own runtime, see [runtime_metrics.md](../../observability/metrics/runtime_metrics.md).

## Supported Monitoring Platforms

**Available now**:
- Self Deployed Prometheus

**Todo**:
- Google Cloud Monitoring
- AWS CloudWatch
- Azure Monitor