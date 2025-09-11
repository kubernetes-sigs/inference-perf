## Run `inference-perf` as a Job in a Kubernetes cluster

This guide explains how to deploy `inference-perf` to a Kubernetes cluster as a job.

### via Helm Chart
Refer to the [guide](./inference-perf/README.md) in `/deploy/inference-perf`.

### via Manual Deployment

#### Setup

`inference-perf` requires all config be configured in a single yaml file and passed via the `-c` flag. When deploying as a job the most straightforward way to pass this value is to create a ConfigMap and then mount the ConfigMap in the Job. Update the `config.yml` as needed then create the ConfigMap by running at the root of this repo:

```bash
kubectl create configmap inference-perf-config --from-file=config.yml
```

**Optional**: Create a Kubernetes Secret that contains the Hugging Face token:

> **Note**: *this step is required for gated models only*

```bash
kubectl create secret generic hf-secret \
    --from-literal=hf_api_token=${HF_TOKEN} \
    --dry-run=client -o yaml | kubectl apply -f -
```

*\* For huggingface authentication, please refer to **“Hugging Face Authentication”** in the section [Run locally](../README.md#run-locally)*

#### Instructions

Apply the job by running the following:
```bash
kubectl apply -f manifests.yaml
```

#### Viewing Results

Currently, inference-perf outputs benchmark results to standard output only. To view the results after the job completes, run:
```bash
kubectl wait --for=condition=complete job/inference-perf && kubectl logs jobs/inference-perf
```
