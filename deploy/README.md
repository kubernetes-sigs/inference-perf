## Run `inference-perf` as a Job in a Kubernetes cluster

This guide explains how to deploy `inference-perf` to a Kubernetes cluster as a job.

### Setup

`inference-perf` requires all config be configured in a single yaml file and passed via the `-c` flag. When deploying as a job the most straightforward way to pass this value is to create a ConfigMap and then mount the ConfigMap in the Job. Update the `config.yml` as needed then create the ConfigMap by running at the root of this repo:

```bash
kubectl create configmap inference-perf-config --from-file=config.yml
```

### Instructions

Apply the job by running the following:
```bash
kubectl apply -f manifests.yaml
```

### Viewing Results

Currently, inference-perf outputs benchmark results to standard output only. To view the results after the job completes, run:
```bash
kubectl wait --for=condition=complete job/inference-perf && kubectl logs jobs/inference-perf
```
