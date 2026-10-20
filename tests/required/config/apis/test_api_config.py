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
    EmbeddingsConfig,
    EmbeddingsEncodingFormat,
    RerankConfig,
    ResponseFormat,
    ResponseFormatType,
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


def test_rerank_api_type_accepted() -> None:
    cfg = APIConfig(type=APIType.Rerank)
    assert cfg.type == APIType.Rerank
    assert cfg.streaming is False


def test_rerank_rejects_streaming() -> None:
    with pytest.raises(ValidationError, match="streaming is not supported for the rerank API"):
        APIConfig(type=APIType.Rerank, streaming=True)


def test_rerank_rejects_response_format() -> None:
    with pytest.raises(ValidationError, match="response_format is not supported for the rerank API"):
        APIConfig(type=APIType.Rerank, response_format=ResponseFormat(type=ResponseFormatType.JSON_OBJECT))


def test_rerank_rejects_streaming_set_from_cli() -> None:
    # CLI overrides are merged into the config before validation, so the check
    # must also catch `--api.type rerank --api.streaming true`.
    with pytest.raises(ValidationError, match="streaming is not supported for the rerank API"):
        read_config(cli_overrides={"api": {"type": "rerank", "streaming": True}})


def test_rerank_config_defaults() -> None:
    cfg = RerankConfig()
    assert cfg.document_count == 10
    assert cfg.route == "/v1/rerank"
    assert cfg.query_field == "query"
    assert cfg.documents_field == "documents"
    assert cfg.top_n is None


def test_rerank_config_read_from_cli() -> None:
    config = read_config(
        cli_overrides={"api": {"type": "rerank", "rerank": {"document_count": 32, "route": "/rerank", "top_n": 5}}}
    )
    assert config.api.rerank == RerankConfig(document_count=32, route="/rerank", top_n=5)


def test_rerank_config_accepts_top_n_zero() -> None:
    # vLLM's rerank schema uses top_n=0 to mean "return all results".
    cfg = RerankConfig(top_n=0)
    assert cfg.top_n == 0


@pytest.mark.parametrize("field", [{"document_count": 0}, {"top_n": -1}])
def test_rerank_config_rejects_invalid_values(field: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        RerankConfig(**field)


def test_rerank_options_rejected_for_other_api_types() -> None:
    with pytest.raises(ValidationError, match="rerank options are only valid when type is 'rerank'"):
        APIConfig(type=APIType.Completion, rerank=RerankConfig(document_count=8))


def test_rerank_config_rejects_route_without_leading_slash() -> None:
    with pytest.raises(ValidationError, match="route must be non-empty and start with '/'"):
        RerankConfig(route="rerank")


def test_rerank_config_rejects_empty_field_names() -> None:
    with pytest.raises(ValidationError, match="query_field must not be empty"):
        RerankConfig(query_field="")
    with pytest.raises(ValidationError, match="documents_field must not be empty"):
        RerankConfig(documents_field="")


def test_rerank_config_rejects_matching_field_names() -> None:
    with pytest.raises(ValidationError, match="query_field and documents_field must differ"):
        RerankConfig(query_field="text", documents_field="text")


@pytest.mark.parametrize("field", ["query_field", "documents_field"])
@pytest.mark.parametrize("reserved", ["model", "top_n"])
def test_rerank_config_rejects_reserved_field_names(field: str, reserved: str) -> None:
    with pytest.raises(ValidationError, match="may not be 'model' or 'top_n'"):
        RerankConfig(**{field: reserved})
