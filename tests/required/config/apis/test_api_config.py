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

from inference_perf.config import APIConfig, APIType, ResponseFormat, ResponseFormatType, read_config


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
