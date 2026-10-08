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
import string
from enum import Enum
from typing import Any, Iterator, Optional

import jmespath
from jmespath.exceptions import JMESPathError
from inference_perf.config.common import StrictBaseModel
from pydantic import Field, model_validator


class APIType(Enum):
    Completion = "completion"
    Chat = "chat"
    AnthropicMessages = "anthropic_messages"
    Embeddings = "embeddings"
    Template = "template"


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


# Values a template body can use, written as ${name}.
_TEMPLATE_PLACEHOLDERS = ("prompt", "max_tokens", "model")


def _template_strings(node: Any) -> Iterator[str]:
    """Every string in a template body, through nested objects and lists."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _template_strings(value)
    elif isinstance(node, list):
        for item in node:
            yield from _template_strings(item)


def _placeholders(text: str, where: str) -> set[str]:
    """The placeholder names in text, after checking that every $ is well formed."""
    template = string.Template(text)
    if not template.is_valid():
        raise ValueError(f"template {where} has an invalid placeholder in '{text}'. Write a literal $ as $$.")
    return set(template.get_identifiers())


def _fill_template(node: Any, values: dict[str, Any]) -> Any:
    """A copy of a template body with every placeholder replaced by its value.

    A string that is only one placeholder takes the value as it is, so
    ${max_tokens} is sent as a number and not as text.
    """
    if isinstance(node, str):
        if node.startswith("${") and node.endswith("}") and node[2:-1] in values:
            return values[node[2:-1]]
        return string.Template(node).substitute(values)
    if isinstance(node, dict):
        return {key: _fill_template(value, values) for key, value in node.items()}
    if isinstance(node, list):
        return [_fill_template(item, values) for item in node]
    return node


class TemplateStreamFraming(Enum):
    SSE = "sse"
    NDJSON = "ndjson"


class TemplateStreamChunks(Enum):
    DELTA = "delta"
    CUMULATIVE = "cumulative"


class TemplateStreamConfig(StrictBaseModel):
    """How the template API reads a streamed response."""

    framing: TemplateStreamFraming = Field(
        default=TemplateStreamFraming.SSE,
        description="How the stream is split into chunks: 'sse' for Server-Sent Events data lines, 'ndjson' for one JSON object per line.",
    )
    chunks: TemplateStreamChunks = Field(
        description="'delta' if each chunk holds only the new text, 'cumulative' if each chunk holds all the text so far."
    )


class TemplateResponseConfig(StrictBaseModel):
    """Where the template API finds the generated text and token counts in a response."""

    text_path: str = Field(
        description="JMESPath expression that selects the generated text in the response body, or in each chunk of a stream. It must select only the generated text, without the prompt."
    )
    input_tokens_path: Optional[str] = Field(
        default=None,
        description="JMESPath expression that selects the prompt token count in the response body, or in the last chunk that has it. Unset counts the prompt with the tokenizer.",
    )
    output_tokens_path: Optional[str] = Field(
        default=None,
        description="JMESPath expression that selects the generated token count in the response body, or in the last chunk that has it. Reported as the server's completion_tokens.",
    )
    stream: Optional[TemplateStreamConfig] = Field(
        default=None, description="How a streamed response is read. Required when streaming is true, and only valid then."
    )

    @model_validator(mode="after")
    def validate_paths(self) -> "TemplateResponseConfig":
        for name, expression in (
            ("text_path", self.text_path),
            ("input_tokens_path", self.input_tokens_path),
            ("output_tokens_path", self.output_tokens_path),
        ):
            if expression is None:
                continue
            try:
                jmespath.compile(expression)
            except JMESPathError as e:
                raise ValueError(f"template response {name} is not a valid JMESPath expression: {e}") from e
        return self


class TemplateConfig(StrictBaseModel):
    """Request template and response paths for the template API (type 'template')."""

    route: str = Field(
        description="Path the request is sent to, appended to the server base URL, e.g. '/generate'. It can use ${model}."
    )
    body: dict[str, Any] = Field(
        description="JSON request body. ${prompt}, ${max_tokens} and ${model} in its string values are filled in for each request."
    )
    ignore_eos: bool = Field(
        default=False,
        description="Declares that the body asks the server to ignore EOS and generate all ${max_tokens} tokens. The body still sets the server's own field for this.",
    )
    response: TemplateResponseConfig = Field(description="Where the generated text and token counts are in the response body.")

    @model_validator(mode="after")
    def validate_template(self) -> "TemplateConfig":
        if not self.route.startswith("/"):
            raise ValueError(f"template route must start with '/', got '{self.route}'")
        route_unknown = sorted(_placeholders(self.route, "route").difference({"model"}))
        if route_unknown:
            raise ValueError(f"template route can only use ${{model}}, got {route_unknown}")
        used: set[str] = set()
        for text in _template_strings(self.body):
            used.update(_placeholders(text, "body"))
        unknown = sorted(used.difference(_TEMPLATE_PLACEHOLDERS))
        if unknown:
            raise ValueError(
                f"template body uses unknown placeholders {unknown}. "
                "The known ones are ${prompt}, ${max_tokens} and ${model}."
            )
        if "prompt" not in used:
            raise ValueError("template body must use ${prompt}")
        # ignore_eos means the server generates the full max_tokens, so the body has to send it.
        if self.ignore_eos and "max_tokens" not in used:
            raise ValueError("template ignore_eos needs the body to use ${max_tokens}")
        return self

    def render_route(self, model: str) -> str:
        """The route with ${model} replaced by the model name."""
        return string.Template(self.route).substitute(model=model)

    def render_body(self, values: dict[str, Any]) -> dict[str, Any]:
        """The request body with every placeholder replaced by its value."""
        body: dict[str, Any] = _fill_template(self.body, values)
        return body


class APIConfig(StrictBaseModel):
    type: APIType = Field(
        default=APIType.Completion,
        description="API endpoint to benchmark: text completion, chat completion, Anthropic messages, embeddings or a request template.",
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
    response_format: Optional[ResponseFormat] = Field(
        default=None, description="Structured output settings sent as the 'response_format' request parameter."
    )
    embeddings: Optional[EmbeddingsConfig] = Field(
        default=None, description="Embeddings request options. Only valid when type is 'embeddings'."
    )
    template: Optional[TemplateConfig] = Field(
        default=None, description="Request template and response paths. Required when type is 'template'."
    )
    session_id_header_key: Optional[str] = Field(
        default=None, description="Header used to send the session ID with each request in multi-turn benchmarks."
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
    def validate_embeddings_options(self) -> "APIConfig":
        # /v1/embeddings returns a single JSON body with no generated text, so it
        # can neither stream nor constrain its output to a schema.
        if self.type == APIType.Embeddings:
            if self.streaming:
                raise ValueError("streaming is not supported for the embeddings API")
            if self.response_format is not None:
                raise ValueError("response_format is not supported for the embeddings API")
        elif self.embeddings is not None:
            raise ValueError("embeddings options are only valid when type is 'embeddings'")
        return self

    @model_validator(mode="after")
    def validate_template_options(self) -> "APIConfig":
        # The template is the whole request body, so the client cannot add
        # response_format to it.
        if self.type == APIType.Template:
            if self.template is None:
                raise ValueError("template options are required when type is 'template'")
            if self.streaming and self.template.response.stream is None:
                raise ValueError("template streaming needs template.response.stream")
            if not self.streaming and self.template.response.stream is not None:
                raise ValueError("template.response.stream is only valid when streaming is true")
            if self.response_format is not None:
                raise ValueError("response_format is not supported for the template API")
        elif self.template is not None:
            raise ValueError("template options are only valid when type is 'template'")
        return self
