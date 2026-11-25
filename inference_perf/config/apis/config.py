# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from enum import Enum
from typing import Any, Optional

from inference_perf.config.common import StrictBaseModel
from pydantic import Field, model_validator


class APIType(Enum):
    Completion = "completion"
    Chat = "chat"
    AnthropicMessages = "anthropic_messages"
    Embeddings = "embeddings"
    Rerank = "rerank"
    SGLangGenerate = "sglang_generate"


class EmbeddingsEncodingFormat(Enum):
    FLOAT = "float"
    BASE64 = "base64"


class ResponseFormatType(Enum):
    JSON_SCHEMA = "json_schema"
    JSON_OBJECT = "json_object"


class ResponseFormat(StrictBaseModel):
    """Configuration for structured output via response_format parameter.

    See vLLM docs: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
    """

    type: ResponseFormatType = Field(
        default=ResponseFormatType.JSON_SCHEMA, description="Structured output mode: a full JSON schema or any JSON object."
    )
    name: str = Field(default="structured_output", description="Name given to the JSON schema in the request payload.")
    json_schema: Optional[dict[str, Any]] = Field(
        default=None, description="JSON schema the model output must conform to when type is 'json_schema'."
    )

    def to_api_format(self) -> dict[str, Any]:
        """Convert to the format expected by vLLM/OpenAI API."""
        if self.type == ResponseFormatType.JSON_OBJECT:
            return {"type": "json_object"}
        # json_schema type
        return {
            "type": "json_schema",
            "json_schema": {
                "name": self.name,
                "schema": self.json_schema,
            },
        }


class EmbeddingsConfig(StrictBaseModel):
    """Request options for the embeddings API (type 'embeddings')."""

    batch_size: int = Field(default=1, ge=1, description="Number of input strings sent in each embeddings request.")
    dimensions: Optional[int] = Field(
        default=None, gt=0, description="Embedding size requested from the server. Unset uses the model's default."
    )
    encoding_format: Optional[EmbeddingsEncodingFormat] = Field(
        default=None, description="Format of the returned embeddings: 'float' or 'base64'. Unset uses the server's default."
    )


class RerankConfig(StrictBaseModel):
    """Request options for the rerank API (type 'rerank')."""

    document_count: int = Field(default=10, ge=1, description="Number of documents scored against the query in each request.")
    route: str = Field(default="/v1/rerank", description="Request path for the rerank endpoint.")
    query_field: str = Field(default="query", description="Request field name carrying the query text.")
    documents_field: str = Field(
        default="documents", description="Request field name carrying the list of candidate documents."
    )
    top_n: Optional[int] = Field(
        default=None,
        ge=0,
        description="Optional top_n sent to the server to limit the number of results returned. 0 means all results.",
    )

    @model_validator(mode="after")
    def validate_field_names(self) -> "RerankConfig":
        if not self.route.startswith("/"):
            raise ValueError("route must be non-empty and start with '/'")
        if not self.query_field:
            raise ValueError("query_field must not be empty")
        if not self.documents_field:
            raise ValueError("documents_field must not be empty")
        if self.query_field == self.documents_field:
            raise ValueError("query_field and documents_field must differ")
        if self.query_field in ("model", "top_n") or self.documents_field in ("model", "top_n"):
            raise ValueError("query_field and documents_field may not be 'model' or 'top_n'")
        return self


class AsyncPollingConfig(StrictBaseModel):
    """Polling options for asynchronous 202-Accepted enqueue endpoints (e.g. Coordinator async-broker)."""

    poll_url_template: str = Field(
        default="/v1/requests/{id}",
        description="URL path (relative to server.base_url) or full URL template polled after a 202 response, with '{id}' replaced by the request ID.",
    )
    id_json_field: str = Field(
        default="id",
        min_length=1,
        description="Dot-separated JSON field in the 202 response body carrying the request ID.",
    )
    response_id_header: Optional[str] = Field(
        default="x-request-id",
        description="Response header fallback used to extract the request ID when id_json_field is absent from the 202 body.",
    )
    poll_interval_sec: float = Field(
        default=0.5,
        gt=0.0,
        description="Seconds between GET status polls while the server answers 202.",
    )
    max_poll_duration_sec: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Maximum seconds spent polling a 202 request before failing with TimeoutError. Defaults to load.request_timeout when unset.",
    )
    exclude_request_headers: list[str] = Field(
        default_factory=lambda: ["x-llm-d-async-mode"],
        description="Request header names (case-insensitive) omitted from GET poll requests while forwarding remaining headers such as tenant and auth.",
    )
    headers: Optional[dict[str, str]] = Field(
        default=None,
        description="Additional HTTP headers sent on GET poll requests.",
    )

    @model_validator(mode="after")
    def validate_template(self) -> "AsyncPollingConfig":
        if "{id}" not in self.poll_url_template:
            raise ValueError("poll_url_template must contain '{id}'")
        if not (
            self.poll_url_template.startswith("/")
            or self.poll_url_template.startswith("http://")
            or self.poll_url_template.startswith("https://")
        ):
            raise ValueError("poll_url_template must start with '/', 'http://', or 'https://'")
        return self


# API types that return a single JSON body with no generated text, and so can
# neither stream nor constrain output to a schema.
_NO_GENERATION_API_TYPES = (APIType.Embeddings, APIType.Rerank)


class APIConfig(StrictBaseModel):
    type: APIType = Field(
        default=APIType.Completion,
        description="API endpoint to benchmark: text completion, chat completion, Anthropic messages, embeddings, rerank or SGLang's native /generate.",
    )
    streaming: bool = Field(
        default=False, description="Stream responses instead of waiting for the full response. Enables TTFT and TPOT metrics."
    )
    headers: Optional[dict[str, str]] = Field(default=None, description="Additional HTTP headers to send with every request.")
    slo_unit: Optional[str] = Field(
        default=None, description="Time unit for SLO header values: 's', 'ms' or 'us'. Defaults to 'ms'."
    )
    slo_tpot_header: Optional[str] = Field(
        default=None,
        description="Request header carrying the per-request TPOT SLO threshold. Defaults to 'x-slo-tpot-<slo_unit>'.",
    )
    slo_ttft_header: Optional[str] = Field(
        default=None,
        description="Request header carrying the per-request TTFT SLO threshold. Defaults to 'x-slo-ttft-<slo_unit>'.",
    )
    slo_request_latency_header: Optional[str] = Field(
        default=None,
        description="Request header carrying the per-request end-to-end latency or deadline SLO threshold. Defaults to 'x-slo-request-latency-<slo_unit>'.",
    )
    async_polling: Optional[AsyncPollingConfig] = Field(
        default=None,
        description="Async 202-Accepted enqueue and GET status polling options (e.g. llm-d Coordinator async-broker enqueue mode).",
    )
    response_format: Optional[ResponseFormat] = Field(
        default=None, description="Structured output settings sent as the 'response_format' request parameter."
    )
    embeddings: Optional[EmbeddingsConfig] = Field(
        default=None, description="Embeddings request options. Only valid when type is 'embeddings'."
    )
    rerank: Optional[RerankConfig] = Field(
        default=None, description="Rerank request options. Only valid when type is 'rerank'."
    )
    session_id_header_key: Optional[str] = Field(
        default=None, description="Header used to send the session ID with each request in multi-turn benchmarks."
    )
    clean_tool_schemas: bool = Field(
        default=False,
        description="Rewrite tool parameter schemas into the subset strict servers accept (e.g. vLLM with xgrammar): drop keys such as 'default' and 'title', collapse anyOf/oneOf, coerce object-valued additionalProperties to false. Changes the prompt; use only when the server rejects the recorded schemas.",
    )
    # Response header carrying a server-assigned session token (e.g. x-session-token
    # from the llm-d-router session affinity plugin). When set, the token received in
    # a session's response is echoed as a request header on subsequent requests of
    # the same session so the router can maintain session affinity.
    session_token_header_key: Optional[str] = Field(
        default=None,
        description="Response header carrying a server-assigned session token, replayed as a request header on later requests of the same session to keep router session affinity.",
    )

    @model_validator(mode="after")
    def validate_no_generation_options(self) -> "APIConfig":
        if self.type in _NO_GENERATION_API_TYPES:
            if self.streaming:
                raise ValueError(f"streaming is not supported for the {self.type.value} API")
            if self.response_format is not None:
                raise ValueError(f"response_format is not supported for the {self.type.value} API")
        # /generate takes structured output inside sampling_params, not as response_format.
        if self.type == APIType.SGLangGenerate and self.response_format is not None:
            raise ValueError("response_format is not supported for the sglang_generate API")
        if self.async_polling is not None and self.streaming:
            raise ValueError("streaming is not supported when async_polling is configured")
        if self.type != APIType.Embeddings and self.embeddings is not None:
            raise ValueError("embeddings options are only valid when type is 'embeddings'")
        if self.type != APIType.Rerank and self.rerank is not None:
            raise ValueError("rerank options are only valid when type is 'rerank'")
        return self
