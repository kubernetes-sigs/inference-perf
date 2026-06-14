# Deepseek-R1

## Configuration
| Kind | Model Server | Model | Provider | Accelerator |
| --- | --- | --- | --- | --- |
| Deployment | vLLM | deepseek-r1 | GKE | GPU H100 |

## Usage

The template can be deployed with the following commands:

```
kustomize build core/lws/vllm/deepseek-r1/gke | kubectl apply -f -
```
