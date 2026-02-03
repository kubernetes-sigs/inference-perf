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

### Via CLI flag

```bash
inference-perf -c config.yml --guided-json schema.json
```

### Via config file

You can also specify the schema directly in the config file under `api.response_format`:

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

## Response Format Types

- `json_schema`: Enforces output to match the provided JSON schema
- `json_object`: Ensures output is valid JSON (less strict)

## vLLM Requirements

Structured output requires vLLM with guided decoding support. See:
https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters-for-chat-api
