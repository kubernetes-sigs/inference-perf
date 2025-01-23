# Design

This document describes the high level design for the tool. It includes the
following components.

## Dataset Preprocessor

Dataset Preprocessor takes in a known dataset like ShareGPT or OpenOrca as the
input and pre-processes them by making sure the prompt length and generation
length are aligned with the user input to support different options like fixed
input / output length tests, variable length tests (larger input / smaller
output and the vice versa). This allows us to support different GenAI use cases
like chat completion, summarization, code completion, etc. depending on the
dataset and the benchmarking user’s inputs.

## Load Generator

Load Generator is the component which generates different traffic patterns based
on user input. This can include a fixed RPS test for a predetermined amount of
time or include a way to generate bursts in traffic or other traffic patterns as
desired for autoscaling and other use cases.

## Request Processor

Request Processor provides a way to support different model servers and their
corresponding request payload with different configurable parameters. This makes
our tool model server agnostic and provides a generic way to benchmark different
model servers and produce apples to apples comparison between them. This
component will also support different protocols like http and grpc and options
like request streaming which is important to produce time to first token (TTFT)
metric.

## Response Processor / Data Collector

Response Processor / Data Collector component allows us to process the response
and measure the actual performance of the model server in terms of request
latency, TPOT, TTFT and throughput.

## Report Generator / Metrics Exporter

Report Generator / Metrics Exporter generates a report based on the data
collected during benchmarking. It can also export the different metrics that we
collected during benchmarking as metrics into Prometheus which can then be
consumed by other monitoring or visualization solutions.

![benchmarking-tool-architecture](./images/design.png)

## Metrics to Collect

The following are the essential metrics that we want to collect using the
benchmarking tool.

*   Throughput
    *   Output tokens / second
    *   Input tokens / second
    *   Requests / second
*   Latency at different percentiles (mean, median, p90, p99)
    *   Time per output token (TPOT)
    *   Inter-token latency (ITL)
    *   Time to first token (TTFT)
    *   Time per request
*   Request metrics (mean, median, p90, p99)
    *   Prompt tokens
    *   Output tokens

Optionally we also want to collect specific accelerator and model server metrics.

*   Accelerator metrics (mean, median, p90, p99)
    *   Accelerator utilization (duty cycle)
    *   Accelerator memory utilization
    *   Accelerator memory bandwidth utilization
    *   Accelerator power usage
*   Model server metrics (mean, median, p90, p99)
    *   Batch size
    *   Queue size
    *   KV cache usage
