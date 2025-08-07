## Run `inference-perf` as a Job in a Kubernetes cluster

This guide explains how to deploy `inference-perf` to a Kubernetes cluster as a job.

### Setup

Running `inference-perf` requires an input file. This should be provided via a Kubernetes ConfigMap. Update the `config.yml` as needed then create the ConfigMap by running at the root of this repo:

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
