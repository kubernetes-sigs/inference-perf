[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub Release](https://img.shields.io/github/v/release/kubernetes-sigs/inference-perf.svg?label=Release)](https://github.com/kubernetes-sigs/inference-perf/releases/latest)
[![PyPI Release](https://img.shields.io/pypi/v/inference-perf.svg?label=PyPI%20Release)](https://pypi.python.org/pypi/inference-perf)
[![Container Image](https://img.shields.io/badge/Container-latest-blue)](https://quay.io/inference-perf/inference-perf)
[![Tests](https://img.shields.io/github/actions/workflow/status/kubernetes-sigs/inference-perf/unit_test.yml?branch=main&label=Tests)](https://github.com/kubernetes-sigs/inference-perf/actions/workflows/unit_test.yml)
[![Join Slack](https://img.shields.io/badge/Join_Slack-blue?logo=slack)](https://kubernetes.slack.com/?redir=%2Fmessages%2Finference-perf)

# Inference Perf

Inference Perf is a GenAI inference performance benchmarking tool that allows you to benchmark and analyze the performance of inference deployments. It is agnostic of model servers and can be used to measure performance and compare different systems apples-to-apples. It is a part of the inference benchmarking and metrics standardization effort in [wg-serving](https://github.com/kubernetes/community/tree/master/wg-serving) which aims to standardize the [benchmark tooling](https://github.com/kubernetes-sigs/wg-serving/tree/main/proposals/013-inference-perf) and the [metrics](https://docs.google.com/document/d/1SpSp1E6moa4HSrJnS4x3NpLuj88sMXr2tbofKlzTZpk/edit?usp=sharing&resourcekey=0-ob5dR-AJxLQ5SvPlA4rdsg) used to measure inference performance across the Kubernetes and model server communities.

## Architecture

![Architecture Diagram](docs/images/architecture.png)

## Key Features

* Highly scalable and can support benchmarking large inference production deployments.
* Reports the key metrics needed to measure LLM performance.
* Supports different real world and synthetic datasets.
* Supports different APIs and supports multiple model servers with enhanced metrics like [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang) and [TGI](https://github.com/huggingface/text-generation-inference).
* Supports benchmarking large deployments with frameworks like [llm-d](https://llm-d.ai/), [Dynamo](https://docs.nvidia.com/dynamo/latest/) and [Inference Gateway](https://gateway-api-inference-extension.sigs.k8s.io/).
* Supports specifying an exact input and output distribution to simulate different scenarios - Gaussian distribution, fixed length, min-max cases are all supported.
* Generates different load patterns and can benchmark specific cases like burst traffic, scaling to saturation and other autoscaling / routing scenarios.
* Supports Multi-turn chat conversations, it can keep context of a series of messages to simulate a conversation. A request in each chat round will keep previouse messages as prefix. see example [config-multi-turn](examples/vllm/config-shared-prefix-multi-turn.yml)
* Supports MultiLoRA traffic splitting to benchmark multiple LoRA adapters simultaneously with configurable traffic distribution and per-adapter reporting. See [loadgen.md](./docs/loadgen.md#multilora-traffic-splitting) for details.

## Roadmap

* Accelerator metrics collection during benchmarks (GPU utilization, memory usage, power usage, etc.).
* Deployment API to help deploy different inference stacks.
* Support for benchmarking non-LLM GenAI use cases.
* Support for different datasets to simulate real world use cases.
* Replaying traffic from production systems.

## Getting Started

### Run locally

- Setup a virtual environment and install inference-perf

    ```
    pip install inference-perf
    ```

- Hugging Face Authentication [**OPTIONAL**]

    > **Optional**: *the step is required for gated models only*
    
    To download tokenizer from the Hugging Face Hub, you need to authenticate. You can do this in one of the following ways:

    1. Using `huggingface-cli login`:
    This will store your token to it's home folder. `~/.cache/huggingface/` by default.

    2. Using Environment Variables:

       You can set the `HF_TOKEN` environment variable:

       ```bash
       export HF_TOKEN=<huggingface-access-token>
       ```

       Alternatively, you can store the token in a file and set the `HF_TOKEN_PATH` environment variable to the path of that file:

       ```bash
       export HF_TOKEN_PATH=<path-to-token-file>
       ```

- Run inference-perf CLI with a configuration file

    ```
    inference-perf --config_file config.yml
    ```

- See more [examples](./examples/)

### Run in a Docker container

- Run the container by mounting your config file.

    ```bash
    docker run -it --rm -v $(pwd)/config.yml:/workspace/config.yml \
    --mount type=bind,src=<path_to_hf_home_dir>,dst=/root/.cache/huggingface/ \
    quay.io/inference-perf/inference-perf
    ```bash
    docker run -it --rm -v $(pwd)/config.yml:/workspace/config.yml \
    --mount type=bind,src=<path_to_hf_home_dir>,dst=/root/.cache/huggingface/ \
    quay.io/inference-perf/inference-perf
    ```

    *\* For huggingface authentication, please refer to **“Hugging Face Authentication”** in the section [Run locally](#run-locally)*

### Run in a Kubernetes cluster

Refer to the [guide](./deploy/README.md) in `/deploy`.

## Configuration

You can configure inference-perf to run with different data generation and load generation configurations today. Please see `config.yml` and examples in `/examples`.

Refer to the [config.md](./docs/config.md) for documentation on all supported configuration options.

### Quickstart via Command Line Flags

You can override any configuration directly from the CLI without using a YAML file. To use CLI flags, please refer to the [CLI Flags Documentation](./docs/cli_flags.md).

Here are some simple examples to run with CLI flags:

- **Random workload with 10 requests / sec**: Run inference-perf with random datagen and input/output length of 1024/512 tokens mean.
    ```
    inference-perf --server.type vllm --server.base_url http://localhost:8000 --data.type random --load.type constant --load.stages '[{"rate": 10, "duration": 10}]' --data.input_distribution.mean 1024 --data.input_distribution.max 2048 --data.output_distribution.mean 512 --data.output_distribution.max 1024 --api.streaming true
    ```

- **Simple Multi-turn Chat**: Run a multi-turn chat benchmark with shared prefix datagen. num_groups * num_prompts_per_group represent distinct user sessions and duration controls the number of turns per session.
    ```
    inference-perf --server.type vllm --server.base_url http://localhost:8000 --data.type shared_prefix --data.shared_prefix.enable_multi_turn_chat true --data.shared_prefix.num_groups 2 --data.shared_prefix.num_prompts_per_group 5 --data.shared_prefix.system_prompt_len 100 --data.shared_prefix.question_len 50 --data.shared_prefix.output_len 50 --load.stages '[{"rate": 10, "duration": 10}]' --api.streaming true
    ```

- **Trace Replay with Azure Dataset**: Replay a real-world trace with Azure format.
    ```
    inference-perf --server.type vllm --server.base_url http://localhost:8000 --load.type trace_replay --load.trace.file AzureLLMInferenceTrace_conv.csv --data.type random --data.trace.file AzureLLMInferenceTrace_conv.csv --load.stages '[{"rate": 1, "duration": 1}]' --api.streaming true
    ```

- **Run dataset with 100 QPS**: Run ShareGPT or Billsum dataset with 100 QPS.
    ```
    inference-perf --server.type vllm --server.base_url http://localhost:8000 --data.type shareGPT --data.path ShareGPT_V3_unfiltered_cleaned_split.json --load.type constant --load.stages '[{"rate": 100, "duration": 60}]' --api.streaming true
    ```

### Datasets

Supported datasets include the following:
- [ShareGPT](./examples/vllm/config.yml) for a real world conversational dataset
- [Synthetic](./examples/vllm/config-synthetic.yml) for specific input / output distributions with Sonnet data
- [Random](./examples/vllm/config-random.yml) for specific input / output distributions with random data
- [SharedPrefix](./examples/vllm/config-shared-prefix.yml) for prefix caching scenarios
- [CNN DailyMail](./docs/config.md#data-generation) for Summarization use case
- [Billsum Conversations](./docs/config.md#data-generation) for long context prefill heavy cases
- [Infinity Instruct](./docs/config.md#data-generation) for long context decode heavy cases
- mock (for testing)

### Load Generators

Multiple load generators are supported:
- Poisson / constant-time load generation to send specific QPS.
- Multi-process load generation for increased concurrency and higher QPS.
- Concurrent load generation to achieve concurrent user load for a system.

Multiple load patterns can be specified:
- Stages with configurable duration and QPS rate along with specific timeouts in between them allows you to simulate different load patterns like burst in traffic, constantly increasing load till hardware saturation, etc.

Load generator reports metrics per stage on the delays between the request schedule time vs the actual send time. Ideally the schedule_delay should be near 0, if not the load generator is failing to meet the desired load. For detailed information on benchmarking at scale and to understand how inference-perf achieves the load target, please refer to [loadgen.md](./docs/loadgen.md)

Example:
```
"load_summary": {
"count": 480,
"schedule_delay": {
    "mean": 0.0033437913275217094,
    "min": -0.0008108859183266759,
    "p10": -2.9846763936802738e-05,
    "median": 0.0010809275845531374,
    "p90": 0.007055185985518622,
    "max": 0.06699507019948214
},
"send_duration": 59.98128472798271,
"requested_rate": 8.0,
"achieved_rate": 8.00249614820385
}
```

### API

OpenAI completion and chat completion APIs are supported. It can be pointed to any endpoints which support these APIs - currently verified against vLLM deployments. Other APIs and model server support can be added easily.

### Metrics

Different latency and throughput metrics to analyze the performance of different LLM workloads are reported. For a definition of the metrics, please refer to [metrics.md](./docs/metrics.md).

When you run `inference-perf`, it displays a summary table in the CLI:

![Metrics Summary](./docs/images/metrics-summary.png)

### Reports

`inference-perf` generates detailed reports in JSON format. For a detailed description of the generated reports and how to read them, please refer to [reports.md](./docs/reports.md).

### Goodput

`inference-perf` supports measuring Goodput, which counts requests that meet specific SLA/SLO constraints. For details on how to configure and measure goodput, please refer to [goodput.md](./docs/goodput.md).

## Analysis

Reports can be analyzed using the following command:

```
inference-perf --analyze <path-to-dir-with-reports>
```

This should generate the following charts (below charts are for example only):

1. QPS vs Latency (TTFT, NTPOT, ITL)

![qps-latency-chart](./docs/images/latency_vs_qps.png)

2. QPS vs Throughput (input tokens / sec, output tokens / sec, total tokens / sec)

![qps-throughput-chart](./docs/images/throughput_vs_qps.png)

3. Latency vs Throughput (output tokens / sec vs TTFT, NTPOT and ITL)

![latency-throughput-chart](./docs/images/throughput_vs_latency.png)

## Testing

inference-perf comes with a suite of end-to-end tests that you can run to ensure that no regressions in functionality or performance was introduced. To run the tests, you have two ways.

### Testing Locally

You can run the end-to-end test suites locally from within the command line.

It is recommended that you have `llm-d-inference-sim` installed locally to ensure that the full test suite is executed. To do so, you can use the Nix flake available locally by running `nix develop`, or run your tests in Docker instead (see below).

To start all end-to-end tests, run:

```sh
pdm run test:e2e
```

### Testing in Docker

Running end-to-end tests in Docker is also a single command:

```sh
pdm run test:e2e:docker
```

This command will automatically build a testable Docker image and run it immediately after.

### Testing using GitHub Actions

Every commit and PR pushed to the `main` or `feature/*` branches will automatically trigger the CI pipelines to run tests using GitHub Actions. This ensures that all test cases are verified automatically before merging any changes.

To debug any failure in testing, the end-to-end test workflow exposes output artifacts, such as debug-level test output logs. To access this, navigate to the workflow run page for the specific commit or PR. For more information, see GitHub's [Downloading workflow artifacts][] page.

[Downloading workflow artifacts]: https://docs.github.com/en/actions/how-tos/manage-workflow-runs/download-workflow-artifacts

## Contributing

Our community meeting is weekly on Thursdays alternating betweem 09:00 and 11:30 PDT ([Zoom Link](https://zoom.us/j/9955436256?pwd=Z2FQWU1jeDZkVC9RRTN4TlZyZTBHZz09), [Meeting Notes](https://docs.google.com/document/d/15XSF8q4DShcXIiExDfyiXxAYQslCmOmO2ARSJErVTak/edit?usp=sharing), [Meeting Recordings](https://www.youtube.com/playlist?list=PL69nYSiGNLP30qNanabU75ayPK7OPNAAS)). 

We currently utilize the [#inference-perf](https://kubernetes.slack.com/?redir=%2Fmessages%2Finference-perf) channel in Kubernetes Slack workspace for communications.

Contributions are welcomed, thanks for joining us!

### Code of conduct

Participation in the Kubernetes community is governed by the [Kubernetes Code of Conduct](code-of-conduct.md).
