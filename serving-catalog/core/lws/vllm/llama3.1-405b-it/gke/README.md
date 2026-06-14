# Llama3.1-405b

## Configuration
| Kind | Model Server | Model | Provider | Accelerator |
| --- | --- | --- | --- | --- |
| Deployment | vLLM | llama3.1-405b-it | GKE | GPU H100 |

## Usage

The template can be deployed with the following commands:

```
kustomize build core/lws/vllm/llama3.1-405b-it/gke | kubectl apply -f -
```
