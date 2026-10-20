# Embeddings Benchmarking

This example benchmarks an embeddings endpoint (`POST /v1/embeddings`) with the
`embeddings` API type.

## Why benchmark embeddings?

Embedding models turn text into vectors for search, retrieval (RAG) and
recommendation. They are usually called with many texts per request, so the
number of inputs per request (the batch size) has a large effect on both
throughput and latency.

An embeddings request generates no tokens, so it is measured differently from
completion or chat:

- **Reported:** request latency, requests per second, and input tokens per
  second (from the server's `usage.prompt_tokens`).
- **Not reported:** TTFT, TPOT, ITL and NTPOT. They measure generated tokens,
  so they are left unset (`null`) rather than reported as 0.
- `streaming` and `response_format` are not supported for embeddings.

## Usage

Configure the embeddings request under `api`:

```yaml
api:
  type: embeddings
  embeddings:
    batch_size: 16          # Input strings per request (default: 1)
    dimensions: 512         # Optional; defaults to the model's embedding size
    encoding_format: float  # Optional: float or base64
```

With the `synthetic` data generator, `input_distribution` sets the length of
each input string, and each request carries `batch_size` of them. No
`output_distribution` is needed. The `mock` data generator also supports
embeddings.

Embeddings are supported by the `vllm`, `sglang` and `mock` server types.

## Running the example

Start a vLLM server with an embedding model:

```bash
vllm serve BAAI/bge-small-en-v1.5
```

Then run the benchmark:

```bash
inference-perf --config_file examples/embeddings/config.yml
```

### With SGLang

Start SGLang in embedding mode with `--is-embedding`:

```bash
python3 -m sglang.launch_server --model-path BAAI/bge-small-en-v1.5 \
  --is-embedding --port 8000 --attention-backend triton
```

`--attention-backend triton` is needed for this model: SGLang's default
FlashInfer backend fails on its small attention heads. Larger embedding models
may not need it.

Then run the benchmark with the server type set to `sglang`:

```bash
inference-perf --config_file examples/embeddings/config.yml --server.type sglang
```

## Sweeping batch sizes

The batch size can be overridden from the command line, so comparing batch
sizes is a loop over the same config:

```bash
for bs in 1 8 32 64; do
  inference-perf --config_file examples/embeddings/config.yml \
    --api.embeddings.batch_size "$bs" \
    --storage.local_storage.path "reports/embeddings-bs$bs"
done
```

Compare `requests_per_sec` and the request latency across runs. Inputs per
second is `requests_per_sec × batch_size`.
