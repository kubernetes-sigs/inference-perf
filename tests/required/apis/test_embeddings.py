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
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from inference_perf.apis import EmbeddingsAPIData, RequestLifecycleMetric, UnaryResponseMetrics
from inference_perf.config import APIConfig, APIType, EmbeddingsConfig, EmbeddingsEncodingFormat
from inference_perf.reportgen.base import compute_request_latency_metrics


def _make_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.count_tokens = lambda text, **kw: max(1, len((text or "").split()))
    return tok


def _make_response(body: dict[str, Any]) -> MagicMock:
    response = MagicMock()
    response.json = AsyncMock(return_value=body)
    return response


_CONFIG = APIConfig(type=APIType.Embeddings)


def test_embeddings_api_type_and_route() -> None:
    data = EmbeddingsAPIData(input="hello")
    assert data.get_api_type() == APIType.Embeddings
    assert data.get_route() == "/v1/embeddings"


def test_embeddings_from_texts_applies_options() -> None:
    options = EmbeddingsConfig(batch_size=2, dimensions=128, encoding_format=EmbeddingsEncodingFormat.FLOAT)
    batch = EmbeddingsAPIData.from_texts(["a", "b"], options)
    assert batch.input == ["a", "b"]
    assert batch.dimensions == 128
    assert batch.encoding_format == EmbeddingsEncodingFormat.FLOAT

    # One text is sent as a plain string; no options means server defaults.
    single = EmbeddingsAPIData.from_texts(["a"], None)
    assert single.input == "a"
    assert single.dimensions is None
    assert single.encoding_format is None


def test_embeddings_from_texts_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="at least one input"):
        EmbeddingsAPIData.from_texts([], None)


@pytest.mark.asyncio
async def test_embeddings_request_body_single_string() -> None:
    data = EmbeddingsAPIData(input="Hello, world!")
    # Generation-only arguments (max_tokens, ignore_eos, streaming) are not sent,
    # and unset optional fields are left out so the server uses its defaults.
    assert await data.to_request_body("test-model", 100, True, False) == {
        "model": "test-model",
        "input": "Hello, world!",
    }


@pytest.mark.asyncio
async def test_embeddings_request_body_batch_with_options() -> None:
    data = EmbeddingsAPIData(
        input=["first text", "second text"],
        dimensions=256,
        encoding_format=EmbeddingsEncodingFormat.BASE64,
    )
    assert await data.to_request_body("test-model", 100, False, False) == {
        "model": "test-model",
        "input": ["first text", "second text"],
        "dimensions": 256,
        "encoding_format": "base64",
    }


@pytest.mark.asyncio
async def test_embeddings_process_response_uses_server_prompt_tokens() -> None:
    data = EmbeddingsAPIData(input=["one two", "three"])
    response = _make_response(
        {
            "object": "list",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]},
                {"object": "embedding", "index": 1, "embedding": [0.4, 0.5, 0.6]},
            ],
            "usage": {"prompt_tokens": 9, "total_tokens": 9},
        }
    )

    info = await data.process_response(response, _CONFIG, _make_tokenizer())

    assert info.request_metrics.text.input_tokens == 9
    assert isinstance(info.response_metrics, UnaryResponseMetrics)
    assert info.response_metrics.output_tokens == 0
    assert info.response_metrics.server_usage == {"prompt_tokens": 9, "total_tokens": 9}
    # The vectors are not kept anywhere on the result.
    assert info.extra_info == {}


@pytest.mark.asyncio
async def test_embeddings_process_response_falls_back_without_server_usage() -> None:
    # Without usage.prompt_tokens, every input in the batch is tokenized client-side.
    data = EmbeddingsAPIData(input=["one two", "three four five"])
    response = _make_response({"data": [{"index": 0, "embedding": [0.1]}, {"index": 1, "embedding": [0.2]}]})

    info = await data.process_response(response, _CONFIG, _make_tokenizer())

    assert info.request_metrics.text.input_tokens == 5
    assert info.response_metrics is not None
    assert info.response_metrics.server_usage is None


@pytest.mark.asyncio
async def test_embeddings_leave_token_latency_metrics_unset() -> None:
    # An embeddings response has no generated tokens, so TTFT, TPOT and ITL do
    # not apply: they must be None, not 0.
    data = EmbeddingsAPIData(input="hello world")
    info = await data.process_response(
        _make_response({"data": [{"index": 0, "embedding": [0.1]}], "usage": {"prompt_tokens": 2}}),
        _CONFIG,
        _make_tokenizer(),
    )
    metric = RequestLifecycleMetric(scheduled_time=0.0, start_time=1.0, end_time=1.5, request_data="{}", info=info, error=None)

    latency = compute_request_latency_metrics(metric)

    assert latency["request_latency"] == 0.5
    assert latency["time_to_first_token"] is None
    assert latency["time_per_output_token"] is None
    assert latency["inter_token_latency"] is None
    assert latency["inter_token_latency_deltas"] == []
