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
"""Validity rules for ``inference_perf.config.apis``."""

import re
from typing import Any

import pytest
from pydantic import ValidationError

from inference_perf.config import (
    APIConfig,
    APIType,
    EmbeddingsConfig,
    EmbeddingsEncodingFormat,
    ResponseFormat,
    ResponseFormatType,
    TemplateConfig,
    TemplateStreamChunks,
    TemplateStreamConfig,
    TemplateStreamFraming,
    read_config,
)


def test_api_config_defaults() -> None:
    cfg = APIConfig()
    assert cfg.type == APIType.Completion
    assert cfg.streaming is False
    assert cfg.response_format is None


def test_response_format_json_schema_is_default() -> None:
    fmt = ResponseFormat(json_schema={"type": "object"})
    assert fmt.type == ResponseFormatType.JSON_SCHEMA
    assert fmt.to_api_format() == {
        "type": "json_schema",
        "json_schema": {
            "name": "structured_output",
            "schema": {"type": "object"},
        },
    }


def test_response_format_custom_name_in_api_format() -> None:
    fmt = ResponseFormat(name="my_schema", json_schema={"type": "object"})
    assert fmt.to_api_format()["json_schema"]["name"] == "my_schema"


def test_response_format_json_object() -> None:
    fmt = ResponseFormat(type=ResponseFormatType.JSON_OBJECT)
    assert fmt.to_api_format() == {"type": "json_object"}


def test_embeddings_api_type_accepted() -> None:
    cfg = APIConfig(type=APIType.Embeddings)
    assert cfg.type == APIType.Embeddings
    assert cfg.streaming is False


def test_embeddings_rejects_streaming() -> None:
    with pytest.raises(ValidationError, match="streaming is not supported for the embeddings API"):
        APIConfig(type=APIType.Embeddings, streaming=True)


def test_embeddings_rejects_response_format() -> None:
    with pytest.raises(ValidationError, match="response_format is not supported for the embeddings API"):
        APIConfig(type=APIType.Embeddings, response_format=ResponseFormat(type=ResponseFormatType.JSON_OBJECT))


def test_embeddings_rejects_streaming_set_from_cli() -> None:
    # CLI overrides are merged into the config before validation, so the check
    # must also catch `--api.type embeddings --api.streaming true`.
    with pytest.raises(ValidationError, match="streaming is not supported for the embeddings API"):
        read_config(cli_overrides={"api": {"type": "embeddings", "streaming": True}})


def test_embeddings_config_defaults() -> None:
    cfg = EmbeddingsConfig()
    assert cfg.batch_size == 1
    assert cfg.dimensions is None
    assert cfg.encoding_format is None


def test_embeddings_config_read_from_cli() -> None:
    config = read_config(
        cli_overrides={"api": {"type": "embeddings", "embeddings": {"batch_size": 16, "encoding_format": "base64"}}}
    )
    assert config.api.embeddings == EmbeddingsConfig(batch_size=16, encoding_format=EmbeddingsEncodingFormat.BASE64)


@pytest.mark.parametrize("field", [{"batch_size": 0}, {"dimensions": 0}])
def test_embeddings_config_rejects_non_positive_values(field: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        EmbeddingsConfig(**field)


def test_embeddings_options_rejected_for_other_api_types() -> None:
    # Options that would be silently ignored are an error, like unknown keys.
    with pytest.raises(ValidationError, match="embeddings options are only valid when type is 'embeddings'"):
        APIConfig(type=APIType.Completion, embeddings=EmbeddingsConfig(batch_size=8))


def _template_options(**overrides: Any) -> dict[str, Any]:
    options: dict[str, Any] = {"route": "/generate", "body": {"text": "${prompt}"}, "response": {"text_path": "text"}}
    options.update(overrides)
    return options


def test_template_api_type_requires_options() -> None:
    with pytest.raises(ValidationError, match="template options are required when type is 'template'"):
        APIConfig(type=APIType.Template)


def test_template_streaming_reads_the_stream_block() -> None:
    options = _template_options(response={"text_path": "text", "stream": {"chunks": "cumulative"}})
    config = read_config(cli_overrides={"api": {"type": "template", "streaming": True, "template": options}})

    assert config.api.template is not None
    assert config.api.template.response.stream == TemplateStreamConfig(
        framing=TemplateStreamFraming.SSE, chunks=TemplateStreamChunks.CUMULATIVE
    )


def test_template_streaming_needs_a_stream_block() -> None:
    with pytest.raises(ValidationError, match="template streaming needs template.response.stream"):
        APIConfig(type=APIType.Template, template=TemplateConfig(**_template_options()), streaming=True)


def test_template_stream_block_needs_streaming() -> None:
    template = TemplateConfig(**_template_options(response={"text_path": "text", "stream": {"chunks": "delta"}}))
    with pytest.raises(ValidationError, match="template.response.stream is only valid when streaming is true"):
        APIConfig(type=APIType.Template, template=template)


@pytest.mark.parametrize(
    ("stream", "field"),
    [
        # A wrong chunks value skews the token counts without an error, so it has no default.
        ({"framing": "sse"}, "chunks"),
        ({"chunks": "partial"}, "chunks"),
        ({"framing": "websocket", "chunks": "delta"}, "framing"),
    ],
)
def test_template_stream_block_rejects_invalid_values(stream: dict[str, Any], field: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        TemplateStreamConfig(**stream)
    assert [error["loc"] for error in exc_info.value.errors()] == [(field,)]


def test_template_rejects_response_format() -> None:
    with pytest.raises(ValidationError, match="response_format is not supported for the template API"):
        APIConfig(
            type=APIType.Template,
            template=TemplateConfig(**_template_options()),
            response_format=ResponseFormat(type=ResponseFormatType.JSON_OBJECT),
        )


def test_template_options_rejected_for_other_api_types() -> None:
    with pytest.raises(ValidationError, match="template options are only valid when type is 'template'"):
        APIConfig(type=APIType.Completion, template=TemplateConfig(**_template_options()))


def test_template_config_read_from_cli() -> None:
    config = read_config(cli_overrides={"api": {"type": "template", "template": _template_options()}})
    assert config.api.template == TemplateConfig(**_template_options())


def test_template_route_can_name_the_model() -> None:
    template = TemplateConfig(**_template_options(route="/v1/models/${model}:predict"))
    assert template.render_route("llama") == "/v1/models/llama:predict"


def test_template_ignore_eos_defaults_to_false() -> None:
    assert TemplateConfig(**_template_options()).ignore_eos is False
    template = TemplateConfig(**_template_options(body={"text": "${prompt}", "n": "${max_tokens}"}, ignore_eos=True))
    assert template.ignore_eos is True


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"route": "generate"}, "template route must start with '/'"),
        ({"route": "/generate/${prompt}"}, "template route can only use ${model}, got ['prompt']"),
        ({"route": "/generate?n=${max_tokens}"}, "template route can only use ${model}, got ['max_tokens']"),
        ({"body": {"text": "${promt}"}}, "template body uses unknown placeholders ['promt']"),
        ({"body": {"text": "fixed"}}, "template body must use ${prompt}"),
        ({"body": {"text": "${prompt} costs $5"}}, "Write a literal $ as $$."),
        ({"ignore_eos": True}, "template ignore_eos needs the body to use ${max_tokens}"),
        ({"response": {"text_path": "choices[0"}}, "template response text_path is not a valid JMESPath expression"),
    ],
)
def test_template_config_rejects_invalid_values(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=re.escape(message)):
        TemplateConfig(**_template_options(**overrides))
