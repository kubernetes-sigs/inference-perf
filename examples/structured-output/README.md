# Structured Output Benchmarking

This example demonstrates how to benchmark LLM inference with structured output
(JSON schema enforcement) using the `response_format` parameter.

## Why Structured Output?

vLLM and other inference servers support guided generation where the output is
constrained to match a JSON schema. This is common in production workloads for:

- Tool calling / function calling
- Structured data extraction
- API responses with defined schemas

Structured output can have performance implications due to the additional
constraint validation during generation, making it important to benchmark.

## Usage

Configure `response_format` in your config file under `api`:

```yaml
api:
  type: chat
  response_format:
    type: json_schema
    name: search_queries  # Optional custom name (default: "structured_output")
    json_schema:
      type: object
      properties:
        query: {type: string}
        intent: {type: string}
      required: [query, intent]
```

Then run with:

```bash
inference-perf -c config.yml
```

See [config.yml](config.yml) and [schema.json](schema.json) for a complete example.

## Response Format Types

- `json_schema`: Enforces output to match the provided JSON schema
- `json_object`: Ensures output is valid JSON (less strict)

## vLLM Requirements

Structured output requires vLLM with guided decoding support. See:
https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters-for-chat-api
