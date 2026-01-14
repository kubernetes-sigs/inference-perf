[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub Release](https://img.shields.io/github/v/release/kubernetes-sigs/inference-perf.svg?label=Release)](https://github.com/kuberentes-sigs/inferece-perf/releases/latest)
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

Multiple load patterns can be specified:
- Stages with configurable duration and QPS along with specific timeouts in between them allows you to simulate different load patterns like burst in traffic, constantly increasing load till hardware saturation, etc.

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

Different latency and throughput metrics to analyze the performance of different LLM workloads are reported. A snippet from an example report is below. For a definition of the metrics, please refer to [metrics.md](./docs/metrics.md).
```
"latency": {
    "request_latency": {
        "mean": 3.31325431142327,
        "min": 1.62129471905064,
        "p10": 1.67609986825846,
        "median": 2.11507539497688,
        "p90": 5.94717199734878,
        "max": 6.30658466403838
    },
    "normalized_time_per_output_token": {
        "mean": 0.104340420636009,
        "min": 0.0506654599703325,
        "p10": 0.0523781208830769,
        "median": 0.0670631669655753,
        "p90": 0.189047570470012,
        "max": 0.20343821496898
    },
    "time_per_output_token": {
        "mean": 0.0836929455635872,
        "min": 0.0517028436646797,
        "p10": 0.0530815053513894,
        "median": 0.0611870964678625,
        "p90": 0.152292036800645,
        "max": 0.17837208439984
    },
    "time_to_first_token": {
        "mean": 0.800974442732916,
        "min": 0.0625283779809251,
        "p10": 0.072068731742911,
        "median": 0.203539535985328,
        "p90": 2.26959549135063,
        "max": 4.46773961000145
    },
    "inter_token_latency": {
        "mean": 0.0836929455635872,
        "min": 0.000007129972800612,
        "p10": 0.0534287681337446,
        "median": 0.0591336835059337,
        "p90": 0.084046097996179,
        "max": 0.614475268055685
    }
},
"throughput": {
    "input_tokens_per_sec": 643.576644186323,
    "output_tokens_per_sec": 32.544923821416,
    "total_tokens_per_sec": 676.121568007739,
    "requests_per_sec": 1.0238155253639
},
"prompt_len": {
    "mean": 628.606060606061,
    "min": 4,
    "p10": 11.4,
    "median": 364,
    "p90": 2427.6,
    "max": 3836
},
"output_len": {
    "mean": 31.7878787878788,
    "min": 30,
    "p10": 31,
    "median": 32,
    "p90": 32,
    "max": 32
}
```

### Reports

Reports are generated in JSON format.
- Per stage reports for individual request rates.
- Summary reports for the overall run.
- Request logs / traces for further analysis.

Model server metrics reports from Prometheus collected during the run is also produced.
- Model server specific metrics like queue size, batch size, latency metrics, etc.
- Supports querying metrics from OSS Prometheus and Google Managed Prometheus.

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

## Contributing

Our community meeting is weekly on Thursdays alternating betweem 09:00 and 11:30 PDT ([Zoom Link](https://zoom.us/j/9955436256?pwd=Z2FQWU1jeDZkVC9RRTN4TlZyZTBHZz09), [Meeting Notes](https://docs.google.com/document/d/15XSF8q4DShcXIiExDfyiXxAYQslCmOmO2ARSJErVTak/edit?usp=sharing), [Meeting Recordings](https://www.youtube.com/playlist?list=PL69nYSiGNLP30qNanabU75ayPK7OPNAAS)). 

We currently utilize the [#inference-perf](https://kubernetes.slack.com/?redir=%2Fmessages%2Finference-perf) channel in Kubernetes Slack workspace for communications.

Contributions are welcomed, thanks for joining us!

### Code of conduct

Participation in the Kubernetes community is governed by the [Kubernetes Code of Conduct](code-of-conduct.md).
