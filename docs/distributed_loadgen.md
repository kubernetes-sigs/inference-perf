# Distributed Load Generator

## Overview
The distributed load generator is designed to scale load testing beyond the limits of a single host. It breaks `inference-perf` into strict components that interact via Redis to provide worker coordination, distributed request/response context tracking, and real-time telemetry. This architecture allows scaling to over 10,000 requests per second.

## Architecture Components
- **CLI (Submitter)**: The local command-line interface used to submit jobs, monitor progress via a Terminal User Interface (TUI), and generate final reports.
- **Redis**: The central message broker and data store. It manages job queuing, task distribution, telemetry, and results collection.
- **Orchestrator**: A cluster-based daemon that manages the job lifecycle, generates specific tasks, and coordinates worker synchronization.
- **Workers**: A pool of load generator instances that execute inference requests against the target model server.
- **Model Server (Simulator)**: The target system being benchmarked (e.g., vLLM, SGLang, or a simulator).

## Job Execution Workflow
1. **Job Submission (CLI)**:
   User runs the CLI in `submit` mode:
   ```bash
   pdm run python inference_perf/main.py -c config.yml --mode submit
   ```
   The config is serialized to JSON and pushed to Redis `job_stream`. Live metrics are monitored via the TUI (unless `--headless` is specified).

2. **Job Orchestration (Orchestrator)**:
   The Orchestrator daemon pulls the job from `job_stream`, primes Redis with prompts, generates tasks with precise scheduled offsets, and pushes them to a task stream. It sets a global start time to synchronize workers.

3. **Load Execution (Workers)**:
   Workers pull tasks from the task stream using a Redis Consumer Group, wait for the scheduled time, and execute requests against the Model Server. Metrics are pushed to `results_stream` and summarized to `telemetry_channel`.

4. **Job Completion and Reporting**:
   The Orchestrator signals completion on the `job_status` channel. The CLI receives the signal, drains the `results_stream` to gather all raw data, and generates final reports.

## How to Use

### Running Locally
To run the distributed load generator locally, you need a Redis instance running (default `localhost:6379`).

1. **Start Redis**: Ensure Redis is running locally.
2. **Start the Orchestrator**:
   ```bash
   pdm run python inference_perf/main.py --mode orchestrator
   ```
3. **Start one or more Workers**:
   ```bash
   pdm run python inference_perf/main.py --mode worker --worker-id worker-1
   ```
4. **Submit a Job**:
   ```bash
   pdm run python inference_perf/main.py -c config.yml --mode submit
   ```

### Running in a Cluster (Kubernetes)
The repository includes Helm charts for deploying the distributed components.

1. **Deploy Components**: Deploy Redis, Orchestrator, and Workers using the Helm chart located in `deploy/inference-perf`.
2. **Submit a Job**: Target the cluster's Redis instance from your local CLI:
   ```bash
   pdm run python inference_perf/main.py -c config.yml --mode submit --redis-host <redis-service-ip>
   ```

## CLI Flags
The following flags support the distributed mode:
- `--mode`: Running mode (`local`, `orchestrator`, `worker`, `tui`, `submit`). Default is `local`.
- `--redis-host`: Redis host address. Default is `localhost`.
- `--redis-port`: Redis port. Default is `6379`.
- `--worker-id`: Worker ID (required in `worker` mode).
- `--headless`: Run without TUI in `submit` mode.
