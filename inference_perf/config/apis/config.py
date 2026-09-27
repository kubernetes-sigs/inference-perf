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
import json
import re
from typing import Any, Optional

import jmespath
from jmespath.exceptions import JMESPathError
from inference_perf.config.common import StrictBaseModel
from pydantic import Field, field_validator, model_validator

_PLACEHOLDER_RE = re.compile(r"\{\{(.*?)\}\}", re.DOTALL)
SUPPORTED_TEMPLATE_VARS = frozenset({"prompt", "max_tokens", "model", "ignore_eos", "stream"})


class APIType(Enum):
    Completion = "completion"
    Chat = "chat"
    AnthropicMessages = "anthropic_messages"
    Template = "template"


def normalize_response_path(expr: str) -> str:
    """Normalize a JMESPath or simple JSONPath expression into JMESPath syntax."""
    stripped = expr.strip()
    if stripped.startswith("$."):
        return stripped[2:]
    if stripped.startswith("$["):
        return stripped[1:]
    return stripped


def compile_response_path(expr: str) -> jmespath.parser.ParsedResult:
    """Compile a JMESPath (or JSONPath-prefixed) expression, raising ValueError if invalid."""
    normalized = normalize_response_path(expr)
    if not normalized:
        raise ValueError(f"Invalid response path expression {expr!r}: expression cannot be empty")
    try:
        return jmespath.compile(normalized)
    except JMESPathError as e:
        raise ValueError(f"Invalid response path expression {expr!r}: {e}") from e


def render_request_template(
    template_str: str,
    *,
    prompt: str,
    max_tokens: int,
    model: str,
    ignore_eos: bool = False,
    stream: bool = False,
) -> dict[str, Any]:
    """Render a JSON body template with request placeholders and parse it as a JSON object."""
    values: dict[str, Any] = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "model": model,
        "ignore_eos": ignore_eos,
        "stream": stream,
    }
    out: list[str] = []
    in_string = False
    escaped = False
    pos = 0

    for match in _PLACEHOLDER_RE.finditer(template_str):
        segment = template_str[pos : match.start()]
        for ch in segment:
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
        out.append(segment)

        var_name = match.group(1).strip()
        if var_name not in values:
            supported = ", ".join(sorted(SUPPORTED_TEMPLATE_VARS))
            raise ValueError(f"Unknown placeholder {var_name!r} in request_template; supported: {supported}")

        val = values[var_name]
        if in_string:
            out.append(json.dumps(str(val))[1:-1])
        else:
            out.append(json.dumps(val))

        pos = match.end()

    out.append(template_str[pos:])

    rendered = "".join(out)
    try:
        parsed = json.loads(rendered)
    except json.JSONDecodeError as e:
        raise ValueError(f"request_template did not render to valid JSON: {e}") from e

    if not isinstance(parsed, dict):
        raise ValueError("request_template must render to a JSON object")
    return parsed


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


class TemplateConfig(StrictBaseModel):
    """Request template and response extraction options for the template API (type 'template')."""

    route: str = Field(
        ...,
        min_length=1,
        description="HTTP endpoint path to POST requests to (e.g. '/generate' or '/v1/completions').",
    )
    request_template: str = Field(
        ...,
        min_length=1,
        description="JSON object template for the request body, with placeholders '{{ prompt }}', '{{ max_tokens }}', '{{ model }}', '{{ ignore_eos }}', and '{{ stream }}'.",
    )
    output_path: str = Field(
        ...,
        min_length=1,
        description="JMESPath or JSONPath expression naming where the generated text lives in the response JSON (e.g. 'generated_text' or 'choices[0].text').",
    )
    input_tokens_path: Optional[str] = Field(
        default=None,
        description="JMESPath or JSONPath expression naming the input token count in the response JSON (e.g. 'usage.prompt_tokens'). Unset falls back to client-side tokenization.",
    )
    output_tokens_path: Optional[str] = Field(
        default=None,
        description="JMESPath or JSONPath expression naming the output token count in the response JSON (e.g. 'usage.completion_tokens'). Unset falls back to client-side tokenization.",
    )

    @field_validator("route")
    @classmethod
    def validate_route(cls, v: str) -> str:
        if not v.startswith("/"):
            raise ValueError("route must start with '/'")
        return v

    @field_validator("request_template")
    @classmethod
    def validate_request_template(cls, v: str) -> str:
        render_request_template(v, prompt="test", max_tokens=16, model="test-model")
        return v

    @field_validator("output_path")
    @classmethod
    def validate_output_path(cls, v: str) -> str:
        compile_response_path(v)
        return v

    @field_validator("input_tokens_path", "output_tokens_path")
    @classmethod
    def validate_optional_token_path(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            compile_response_path(v)
        return v


class APIConfig(StrictBaseModel):
    type: APIType = Field(
        default=APIType.Completion,
        description="API endpoint to benchmark: text completion, chat completion, Anthropic messages, or template.",
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
    template: Optional[TemplateConfig] = Field(
        default=None,
        description="Request template and response extraction options. Required when type is 'template'.",
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
    def validate_template_options(self) -> "APIConfig":
        if self.type == APIType.Template:
            if self.template is None:
                raise ValueError("template options are required when type is 'template'")
            if self.streaming:
                raise ValueError("streaming is not supported for the template API")
            if self.response_format is not None:
                raise ValueError("response_format is not supported for the template API")
        elif self.template is not None:
            raise ValueError("template options are only valid when type is 'template'")
        return self
