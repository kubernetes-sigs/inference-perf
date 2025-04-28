# Inference Perf

The Inference Perf project aims to provide GenAI inference performance benchmarking tool. It came out of [wg-serving](https://github.com/kubernetes/community/tree/master/wg-serving) and is sponsored by [SIG Scalability](https://github.com/kubernetes/community/blob/master/sig-scalability/README.md#inference-perf). See the [proposal](https://github.com/kubernetes-sigs/wg-serving/tree/main/proposals/013-inference-perf) for more info.

## Status

This project is currently in development.

## Getting Started

### Run locally

- Setup a virtual environment and install inference-perf

    ```
    pip install .
    ```

- Run inference-perf CLI with a configuration file

    ```
    inference-perf --config_file config.yml
    ```

- See more [examples](./examples/)

### Run in a Docker container

- Build the container

    ```
    docker build -t inference-perf .
    ```

- Run the container

    ```
    docker run -it --rm -v $(pwd)/config.yml:/workspace/config.yml inference-perf

    ```

# Run in Kubernetes cluster

Refer to [README](./deploy/README.md) in `/deploy`.

## Contributing

Our community meeting is weekly on Thursdays alternating betweem 09:00 and 11:30 PDT ([Zoom Link](https://zoom.us/j/9955436256?pwd=Z2FQWU1jeDZkVC9RRTN4TlZyZTBHZz09), [Meeting Notes](https://docs.google.com/document/d/15XSF8q4DShcXIiExDfyiXxAYQslCmOmO2ARSJErVTak/edit?usp=sharing), [Meeting Recordings](https://www.youtube.com/playlist?list=PL69nYSiGNLP30qNanabU75ayPK7OPNAAS)). 

We currently utilize the [#wg-serving](https://kubernetes.slack.com/?redir=%2Fmessages%2Fwg-serving) Slack channel for communications.

Contributions are welcomed, thanks for joining us!

### Code of conduct

Participation in the Kubernetes community is governed by the [Kubernetes Code of Conduct](code-of-conduct.md).
