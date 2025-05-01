## Run `inference-perf` as a Job in a Kubernetes cluster

This guide explains how to deploy `inference-perf` to a Kubernetes cluster as a job.

> [!NOTE]
> There is currently no support for persisting output reports, all outputs are currently printed to standard output, please refer to issue [#59](https://github.com/kubernetes-sigs/inference-perf/issues/59)

### Setup

Since public container images are not actively being published, you'll need to build the `inference-perf` image yourself. Follow the [official guide](https://github.com/kubernetes-sigs/inference-perf?tab=readme-ov-file#run-in-a-docker-container) to build the container.

Once built, push the image to your preferred container registry:
- [Artifact Registry (Google Cloud)](https://cloud.google.com/artifact-registry/docs/docker/pushing-and-pulling)
- [Docker Hub](https://docs.docker.com/get-started/introduction/build-and-push-first-image/)

Take note of the image name once successfully pushed, replace `<your image here>` in `manifests.yaml` with this image name.

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