# Rerank Benchmarking

This example benchmarks a rerank endpoint (`POST /v1/rerank`) with the
`rerank` API type.

## Why benchmark rerank?

Reranking scores a query against a list of candidate documents, typically as
the second stage of a RAG pipeline after a cheaper retrieval step. It's
usually called with one query against 10-100 documents per request, so the
document count has a large effect on both throughput and latency.

A rerank request generates no tokens, so it is measured differently from
completion or chat:

- **Reported:** request latency, requests per second, and input tokens per
  second. Input tokens prefer the server's reported usage (`usage.prompt_tokens`,
  falling back to `usage.total_tokens` when a server reports only that), and
  fall back further to tokenizing the query and each document client-side when
  a response has no usage at all.
- **Not reported:** TTFT, TPOT, ITL and NTPOT. They measure generated tokens,
  so they are left unset (`null`) rather than reported as 0.
- `streaming` and `response_format` are not supported for rerank.

## Usage

Configure the rerank request under `api`:

```yaml
api:
  type: rerank
  rerank:
    document_count: 20        # Documents scored per query (default: 10)
    route: /v1/rerank         # Optional; vLLM also serves /rerank and /v2/rerank
    query_field: query        # Optional; request field name for the query
    documents_field: documents # Optional; request field name for the documents list
    top_n: 5                  # Optional; ask the server to return only the top N results
```

`route`, `query_field` and `documents_field` exist because rerank has no
single wire format: vLLM, Cohere and Jina all use slightly different request
shapes. `/v1/rerank` with `query`/`documents` fields is vLLM's default; point
`route` and the field names at a different server's shape as needed.

With the `synthetic` data generator, `input_distribution` sets the length of
the query and of each document; each request carries one query plus
`document_count` documents, so `1 + document_count` lengths are sampled per
request. No `output_distribution` is needed. The `mock` data generator also
supports rerank.

Rerank is supported by the `vllm` and `mock` server types.

## Running the example

Start a vLLM server with a reranker model:

```bash
vllm serve BAAI/bge-reranker-base
```

Then run the benchmark:

```bash
inference-perf --config_file examples/rerank/config.yml
```

## Sweeping document counts

The document count can be overridden from the command line, so comparing
batch shapes is a loop over the same config:

```bash
for docs in 10 25 50 100; do
  inference-perf --config_file examples/rerank/config.yml \
    --api.rerank.document_count "$docs" \
    --storage.local_storage.path "reports/rerank-docs$docs"
done
```

Compare `requests_per_sec` and the request latency across runs. Documents
scored per second is `requests_per_sec × document_count`.
