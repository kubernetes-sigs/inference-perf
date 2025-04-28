## `inference-perf` as a Kubernetes Job

This guide explains how to deploy `inference-perf` to a Kubernetes cluster as a job.

> [!NOTE]
> There is currently no support for persisting output reports, all outputs are currently printed to standard output, please refer to issue [#59](https://github.com/kubernetes-sigs/inference-perf/issues/59)
### Setup

Since public container images are not currently published, you'll need to build the `inference-perf` image yourself. Follow the [official guide](https://github.com/kubernetes-sigs/inference-perf?tab=readme-ov-file#run-in-a-docker-container) to build the container.

Once built, push the image to your preferred container registry:
- [Artifact Registry (Google Cloud)](https://cloud.google.com/artifact-registry/docs/docker/pushing-and-pulling)
- [Docker Hub](https://docs.docker.com/get-started/introduction/build-and-push-first-image/)

Take note of the image name once successfully pushed, it should look something like `<your-artifact-registry-region>-docker.pkg.dev/<your-project-name>/<your-artifact-registry-name>/inference-perf:latest`, in `manifests.yaml` replace `<your image here>` with this image name.

`inference-perf` requires an input configuration file. This should be provided via a Kubernetes ConfigMap. You can create the ConfigMap using:

```bash
kubectl create configmap inference-perf-config --from-file=config.yml
```

### Instructions

Apply the job by running the following:
```
kubectl apply -f manifests.yaml
```

### Viewing Results

Currently, inference-perf outputs benchmark results to standard output only. To view the results after the job completes, run:
```
kubectl logs jobs/inference-perf
```