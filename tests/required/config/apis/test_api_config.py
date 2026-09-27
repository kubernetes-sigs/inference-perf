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

import pytest
from pydantic import ValidationError

from inference_perf.config import (
    APIConfig,
    APIType,
    ResponseFormat,
    ResponseFormatType,
    TemplateConfig,
    read_config,
)


_VALID_TEMPLATE = TemplateConfig(
    route="/generate",
    request_template='{"model": {{ model }}, "prompt": {{ prompt }}, "max_tokens": {{ max_tokens }}}',
    output_path="generated_text",
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


def test_template_api_type_accepted() -> None:
    cfg = APIConfig(type=APIType.Template, template=_VALID_TEMPLATE)
    assert cfg.type == APIType.Template
    assert cfg.streaming is False
    assert cfg.template == _VALID_TEMPLATE


def test_template_requires_template_options() -> None:
    with pytest.raises(ValidationError, match="template options are required when type is 'template'"):
        APIConfig(type=APIType.Template)


def test_template_rejects_streaming() -> None:
    with pytest.raises(ValidationError, match="streaming is not supported for the template API"):
        APIConfig(type=APIType.Template, streaming=True, template=_VALID_TEMPLATE)


def test_template_rejects_response_format() -> None:
    with pytest.raises(ValidationError, match="response_format is not supported for the template API"):
        APIConfig(
            type=APIType.Template,
            template=_VALID_TEMPLATE,
            response_format=ResponseFormat(type=ResponseFormatType.JSON_OBJECT),
        )


def test_template_rejects_streaming_set_from_cli() -> None:
    with pytest.raises(ValidationError, match="streaming is not supported for the template API"):
        read_config(
            cli_overrides={
                "api": {
                    "type": "template",
                    "streaming": True,
                    "template": {
                        "route": "/generate",
                        "request_template": '{"prompt": {{ prompt }}}',
                        "output_path": "generated_text",
                    },
                }
            }
        )


def test_template_config_read_from_cli() -> None:
    config = read_config(
        cli_overrides={
            "api": {
                "type": "template",
                "template": {
                    "route": "/generate",
                    "request_template": '{"prompt": {{ prompt }}, "max_tokens": {{ max_tokens }}}',
                    "output_path": "choices[0].text",
                    "input_tokens_path": "usage.prompt_tokens",
                    "output_tokens_path": "usage.completion_tokens",
                },
            }
        }
    )
    assert config.api.template == TemplateConfig(
        route="/generate",
        request_template='{"prompt": {{ prompt }}, "max_tokens": {{ max_tokens }}}',
        output_path="choices[0].text",
        input_tokens_path="usage.prompt_tokens",
        output_tokens_path="usage.completion_tokens",
    )


def test_template_options_rejected_for_other_api_types() -> None:
    with pytest.raises(ValidationError, match="template options are only valid when type is 'template'"):
        APIConfig(type=APIType.Completion, template=_VALID_TEMPLATE)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (
            {"route": "generate", "request_template": '{"prompt": {{ prompt }}}', "output_path": "text"},
            "route must start with '/'",
        ),
        (
            {"route": "/generate", "request_template": '{"prompt": {{ unknown_var }}}', "output_path": "text"},
            "Unknown placeholder 'unknown_var'",
        ),
        (
            {"route": "/generate", "request_template": "not-json", "output_path": "text"},
            "did not render to valid JSON",
        ),
        (
            {"route": "/generate", "request_template": "[1, 2, 3]", "output_path": "text"},
            "must render to a JSON object",
        ),
        (
            {"route": "/generate", "request_template": '{"prompt": {{ prompt }}}', "output_path": "choices[0"},
            "Invalid response path expression",
        ),
        (
            {
                "route": "/generate",
                "request_template": '{"prompt": {{ prompt }}}',
                "output_path": "text",
                "input_tokens_path": "usage.[invalid",
            },
            "Invalid response path expression",
        ),
    ],
)
def test_template_config_validation_errors(kwargs: dict[str, str], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        TemplateConfig(**kwargs)
