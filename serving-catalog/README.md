# Serving Catalog

This repository is a catalog of Kustomize templated yamls for deploying inference workloads.

## Contents

A table of configurations in this repository are available in [catalog.md](./catalog.md).

## Purpose

The purpose of this catalog is to twofold:

- Provide working examples for popular model servers (e.g., vllm, tgi, jetstream, triton/tensorrt) and open models (llama, gemma, mistral, mixtral). This offers a one stop shop:
    - for beginners and evaluators, reducing the barrier to entry. 
    - for reference configurations for production deployments.

- Identify common parameters that users frequently need to tweak and vary, and highlight differences of cloud provider specific configurations (e.g., how to request specific types of accelerators). This will help inform:
    - the design of a potential future domain-specific LLM serving workload API.
    - changes in upstream k8s to cater for these variations.


## Principles

- Templatized so it is easy to parameterize. The catalog uses kustomize for that purpose.
- The catalog should include full working examples. An entry should include full deployment yaml and clear instructions on how to obtain the model server container and load the model, and optionally HPA and monitoring.


**WARNING / STABILITY**
The catalog may change without notice. There is no api guarantee on these templates, structure or behavior, although we intend them to be correct.

## Contributions

All known working configurations are welcome. New top level directories should be created under serving-catalog for new frameworks. The core directory is meant to house examples using minimal k8s based mechanism.

It is recommended to follow the kustomize base/overlay structure `serving-catalog/{platform}/{crd}/{model_server}/{model}/{provider}` shown in core. Provider specific references should be placed in provider specific directories. Components can be used to share common configurations across overlays.

## Information and Usage

Kustomize overlays and components are leveraged to share common configurations across deployment paradigms, model servers, and models. The directory structure reflects this hierarchy:

```
serving-catalog/
├── core
│   ├── deployment
│   │   ├── base
│   │   ├── components
│   │   │   └── gke
│   │   │       └── resources
│   │   │           ├── gpu
│   │   │           │   ├── 1-L4
│   │   │           │   └── 2-L4
│   │   │           └── tpu
│   │   │               └── v5e-2x4
│   │   ├── jetstream
│   │   │   ├── base
│   │   │   └── gemma-7b-it
│   │   │       ├── base
│   │   │       └── gke
│   │   └── vllm
│   │       ├── base
│   │       └── gemma-2b
│   │           ├── base
│   │           └── gke
├── kserve
│   ├── inferenceservice
│   └── servingruntime
```

Where an inference deployment can be deployed using a command like:

```
kustomize build core/deployment/vllm/gemma-2b | kubectl apply -f -
```

For models and model servers that require a model load stage, these commands kustomize and deploy a Job, waits for it to complete and then deploys the inference server.

```
kustomize build core/deployment/jetstream/gemma-7b-it | kubectl apply -f - --selector prerequisite=model-load &&
kubectl wait --for=condition=complete --timeout=1000s job/gemma-7b-it-jetstream-data-loader &&
kustomize build core/deployment/jetstream/gemma-7b-it | kubectl apply -f - --selector app=gemma-7b-it-jetstream-inference-server
```
