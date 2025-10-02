# Inference Performance Metrics Definition

This document outlines the key metrics used for evaluating performance, their definition and how they are calculated.

## Throughput

| Metric | Formula | Unit | Used For
| :--- | :--- | :--- | :---
| **Output tokens / second** | `total output tokens / time in seconds` | tokens / second | Calculating output token throughput of the inference server
| **Input tokens / second** | `total input tokens / time in seconds` | tokens / second | Calculating input token throughput of the inference server
| **Requests / second** | `total requests completed / time in seconds` | qps | Calculating request throughput of the inference server

---

## Latency

| Metric | Formula | Unit | Used For
| :--- | :--- | :--- | :---
| **Time per request (e2e request latency)** | `request completion time - request send time` | seconds per request | Calculating how long a request takes to complete
| **Time to first token (TTFT)** | `time first non empty output token received - request send time` | ms | Calculating the time it takes for the user to receive the first token from the response
| **Time per output token (TPOT)** | `(e2e request latency - ttft ) / (output tokens - 1)` | ms per output token | Calculating the average time it takes for the user to receive successive tokens after the first token
| **Normalized time per output token** | `e2e request latency / output tokens` | ms per output token | Normalizing the request latency at the output token level for comparing different use cases
| **Inter Token Latency (ITL)** | `time between output token generation within a request` | ms per output token | Calculating the time it takes for the user to receive successive tokens after the first token, but at a more granular level than TPOT which averages the token latency within a request

---

## Price/Performance

| Metric | Formula | Unit | Used For
| :--- | :--- | :--- | :---
| **$ per million output tokens*** | `((accelerator $ / second) / (output tokens / second)) * 1M` | $ | Calculating the cost to serve million output tokens
| **$ per million input tokens*** | `((accelerator $ / second) / (input tokens / second)) * 1M` | $ | Calculating the cost to serve million input tokens
| **Throughput / $** | `(output tokens / second) / (accelerator $ / second)` | million output tokens | Calculating the performance to price ratio to get the throughput we are able to achieve for the cost spent

*\*Note: input and output token cost might need to be divided in mixed-batching cases since they are handled together by the server, using some factor like 1:4 for cost to generate input vs output tokens.*
