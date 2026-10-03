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

from inference_perf.apis import RequestLifecycleMetric, RerankAPIData, UnaryResponseMetrics
from inference_perf.config import APIConfig, APIType, RerankConfig
from inference_perf.reportgen.base import compute_request_latency_metrics


def _make_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.count_tokens = lambda text, **kw: max(1, len((text or "").split()))
    return tok


def _make_response(body: dict[str, Any]) -> MagicMock:
    response = MagicMock()
    response.json = AsyncMock(return_value=body)
    return response


_CONFIG = APIConfig(type=APIType.Rerank)


def test_rerank_api_type_and_default_route() -> None:
    data = RerankAPIData(query="what is a cat", documents=["a cat is a mammal"])
    assert data.get_api_type() == APIType.Rerank
    assert data.get_route() == "/v1/rerank"


def test_rerank_from_query_and_documents_uses_configured_route_and_fields() -> None:
    options = RerankConfig(route="/rerank", query_field="q", documents_field="docs", top_n=5)
    data = RerankAPIData.from_query_and_documents("what is a cat", ["doc a", "doc b"], options)

    assert data.query == "what is a cat"
    assert data.documents == ["doc a", "doc b"]
    assert data.get_route() == "/rerank"
    assert data.query_field == "q"
    assert data.documents_field == "docs"
    assert data.top_n == 5

    # No options means server defaults: the vLLM route and field names.
    defaults = RerankAPIData.from_query_and_documents("q", ["doc"], None)
    assert defaults.get_route() == "/v1/rerank"
    assert defaults.query_field == "query"
    assert defaults.documents_field == "documents"
    assert defaults.top_n is None


def test_rerank_from_query_and_documents_rejects_empty_documents() -> None:
    with pytest.raises(ValueError, match="at least one document"):
        RerankAPIData.from_query_and_documents("query", [], None)


@pytest.mark.asyncio
async def test_rerank_request_body_uses_default_field_names() -> None:
    data = RerankAPIData(query="what is a cat", documents=["a cat is a mammal", "dogs bark"])
    # Generation-only arguments (max_tokens, ignore_eos, streaming) are not sent,
    # and top_n is left out when unset so the server uses its default.
    assert await data.to_request_body("test-model", 100, True, False) == {
        "model": "test-model",
        "query": "what is a cat",
        "documents": ["a cat is a mammal", "dogs bark"],
    }


@pytest.mark.asyncio
async def test_rerank_request_body_uses_configured_field_names_and_top_n() -> None:
    data = RerankAPIData(
        query="what is a cat",
        documents=["a cat is a mammal"],
        query_field="q",
        documents_field="docs",
        top_n=3,
    )
    assert await data.to_request_body("test-model", 100, False, False) == {
        "model": "test-model",
        "q": "what is a cat",
        "docs": ["a cat is a mammal"],
        "top_n": 3,
    }


@pytest.mark.asyncio
async def test_rerank_process_response_prefers_server_prompt_tokens() -> None:
    data = RerankAPIData(query="q", documents=["doc a", "doc b"])
    response = _make_response(
        {
            "results": [{"index": 0, "relevance_score": 0.9}, {"index": 1, "relevance_score": 0.1}],
            "usage": {"prompt_tokens": 12, "total_tokens": 12},
        }
    )

    info = await data.process_response(response, _CONFIG, _make_tokenizer())

    assert info.request_metrics.text.input_tokens == 12
    assert isinstance(info.response_metrics, UnaryResponseMetrics)
    assert info.response_metrics.output_tokens == 0
    assert info.response_metrics.server_usage == {"prompt_tokens": 12, "total_tokens": 12}
    # The ranking results are not kept anywhere on the result.
    assert info.extra_info == {}


@pytest.mark.asyncio
async def test_rerank_process_response_falls_back_to_total_tokens() -> None:
    # Some vLLM versions/paths report only total_tokens (no generated tokens on a
    # rerank response, so total_tokens is the prompt-token count).
    data = RerankAPIData(query="q", documents=["doc a"])
    response = _make_response({"results": [{"index": 0}], "usage": {"total_tokens": 7}})

    info = await data.process_response(response, _CONFIG, _make_tokenizer())

    assert info.request_metrics.text.input_tokens == 7
    assert info.response_metrics is not None
    assert info.response_metrics.server_usage == {"total_tokens": 7}


@pytest.mark.asyncio
async def test_rerank_process_response_falls_back_to_tokenizer_without_server_usage() -> None:
    # query="one two" (2 tokens) scored against two documents ("three four" = 2
    # tokens, "five" = 1 token). vLLM scores each query/document pair separately
    # and sums usage across them, so the query is counted once per document:
    # (2 + 2) + (2 + 1) = 7, not tokens(query) + sum(tokens(doc)) = 2 + 3 = 5.
    data = RerankAPIData(query="one two", documents=["three four", "five"])
    response = _make_response({"results": [{"index": 0}, {"index": 1}]})

    info = await data.process_response(response, _CONFIG, _make_tokenizer())

    assert info.request_metrics.text.input_tokens == 7
    assert info.response_metrics is not None
    assert info.response_metrics.server_usage is None


@pytest.mark.asyncio
async def test_rerank_leaves_token_latency_metrics_unset() -> None:
    # A rerank response has no generated tokens, so TTFT, TPOT and ITL do not
    # apply: they must be None, not 0.
    data = RerankAPIData(query="q", documents=["doc a"])
    info = await data.process_response(
        _make_response({"results": [{"index": 0}], "usage": {"prompt_tokens": 4}}),
        _CONFIG,
        _make_tokenizer(),
    )
    assert info.response_metrics is not None
    assert info.response_metrics.output_tokens == 0

    metric = RequestLifecycleMetric(scheduled_time=0.0, start_time=1.0, end_time=1.5, request_data="{}", info=info, error=None)
    latency = compute_request_latency_metrics(metric)

    assert latency["request_latency"] == 0.5
    assert latency["normalized_time_per_output_token"] is None
    assert latency["time_to_first_token"] is None
    assert latency["time_per_output_token"] is None
    assert latency["inter_token_latency"] is None
    assert latency["inter_token_latency_deltas"] == []
