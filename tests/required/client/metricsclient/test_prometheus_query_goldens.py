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
"""Golden PromQL corpus: pins the exact queries every backend declares.

A wrong metric name or label filter never fails a benchmark run; Prometheus
returns an empty result and the report silently loses that column. These
goldens make the declared query surface reviewable: renaming a metric,
changing a filter, or reshaping a query shows up as a golden-file diff next
to the code change that caused it.

Regenerate with UPDATE_QUERY_GOLDENS=1 pytest <this file>. The run fails
after rewriting so a stale golden can never pass CI silently.
"""

import os
from pathlib import Path
from typing import List, Type
from unittest.mock import MagicMock, patch

import pytest

from inference_perf.client.modelserver.metrics import (
    BaseMetrics,
    CounterMetric,
    CounterResult,
    GaugeMetric,
    GaugeResult,
    HistogramMetric,
    HistogramResult,
)
from inference_perf.client.modelserver.openai_client import openAIModelServerClient
from inference_perf.client.modelserver.sglang_client import SGlangModelServerClient
from inference_perf.client.modelserver.tgi_client import TGImodelServerClient
from inference_perf.client.modelserver.vllm_client import vLLMModelServerClient
from inference_perf.config import APIConfig, APIType

GOLDEN_DIR = Path(__file__).parent / "goldens"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DURATION = 60.0


def make_client(client_cls: Type[openAIModelServerClient]) -> openAIModelServerClient:
    with patch("inference_perf.client.modelserver.openai_client.CustomTokenizer"):
        return client_cls(
            metrics_collector=MagicMock(),
            api_config=APIConfig(type=APIType.Completion, streaming=False),
            uri="http://localhost:8000",
            model_name=MODEL,
            tokenizer_config=None,
            max_tcp_connections=4,
            additional_filters=[],
        )


def render_queries(metadata: BaseMetrics) -> List[str]:
    """One line per query, in collection order: '<target_field>\\t<PromQL>'."""
    return [f"{field}\t{query}" for field, metric in metadata for query in metric.get_queries(DURATION, metadata.filters)]


@pytest.mark.parametrize(
    "backend,client_cls",
    [
        ("vllm", vLLMModelServerClient),
        ("sglang", SGlangModelServerClient),
        ("tgi", TGImodelServerClient),
    ],
)
def test_backend_queries_match_golden(backend: str, client_cls: Type[openAIModelServerClient]) -> None:
    rendered = render_queries(make_client(client_cls).get_prometheus_metric_metadata())
    golden_path = GOLDEN_DIR / f"{backend}_queries.txt"
    if os.environ.get("UPDATE_QUERY_GOLDENS"):
        golden_path.write_text("\n".join(rendered) + "\n")
        pytest.fail(f"regenerated {golden_path}; verify the diff, commit it, and re-run without UPDATE_QUERY_GOLDENS")
    assert rendered == golden_path.read_text().splitlines()


def test_gauge_queries_and_parse() -> None:
    metric = GaugeMetric("vllm:num_requests_waiting")
    assert metric.get_queries(60.0, "model_name='m'") == [
        "avg_over_time(vllm:num_requests_waiting{model_name='m'}[60s])",
        "quantile_over_time(0.5, vllm:num_requests_waiting{model_name='m'}[60s])",
        "quantile_over_time(0.9, vllm:num_requests_waiting{model_name='m'}[60s])",
        "quantile_over_time(0.99, vllm:num_requests_waiting{model_name='m'}[60s])",
    ]
    assert metric.parse([1.0, 2.0, 3.0, 4.0]) == GaugeResult(avg=1.0, median=2.0, p90=3.0, p99=4.0)


def test_counter_queries_and_parse() -> None:
    metric = CounterMetric("vllm:prompt_tokens")
    assert metric.get_queries(60.0, "model_name='m'") == [
        "sum(increase(vllm:prompt_tokens_total{model_name='m'}[60s]) or increase(vllm:prompt_tokens{model_name='m'}[60s]))",
        "avg_over_time((rate(vllm:prompt_tokens_total{model_name='m'}[60s])"
        " or rate(vllm:prompt_tokens{model_name='m'}[60s]))[60s:60s])",
        "sum(rate(vllm:prompt_tokens_total{model_name='m'}[60s]) or rate(vllm:prompt_tokens{model_name='m'}[60s]))",
    ]
    assert metric.parse([600.0, 10.0, 10.0]) == CounterResult(total=600.0, avg=10.0, per_second=10.0)


def test_counter_declared_with_total_suffix_is_not_double_suffixed() -> None:
    metric = CounterMetric("sglang:prompt_tokens_total")
    assert (
        metric.get_queries(60.0, "")[2] == "sum(rate(sglang:prompt_tokens_total{}[60s]) or rate(sglang:prompt_tokens{}[60s]))"
    )


def test_counter_over_histogram_series_keeps_single_leg() -> None:
    """`_count`/`_sum`/`_bucket` series can never carry a `_total` suffix, so a counter over
    one (e.g. sglang's requests) must not select a nonexistent `_count_total` leg."""
    metric = CounterMetric("sglang:e2e_request_latency_seconds_count")
    assert metric.get_queries(60.0, "")[0] == "sum(increase(sglang:e2e_request_latency_seconds_count{}[60s]))"


def test_histogram_queries_and_parse() -> None:
    metric = HistogramMetric("vllm:e2e_request_latency_seconds")
    f = "model_name='m'"
    assert metric.get_queries(60.0, f) == [
        f"sum(rate(vllm:e2e_request_latency_seconds_sum{{{f}}}[60s])) / (sum(rate(vllm:e2e_request_latency_seconds_count{{{f}}}[60s])) > 0)",
        f"histogram_quantile(0.5, sum(rate(vllm:e2e_request_latency_seconds_bucket{{{f}}}[60s])) by (le))",
        f"histogram_quantile(0.9, sum(rate(vllm:e2e_request_latency_seconds_bucket{{{f}}}[60s])) by (le))",
        f"histogram_quantile(0.99, sum(rate(vllm:e2e_request_latency_seconds_bucket{{{f}}}[60s])) by (le))",
        f"sum(rate(vllm:e2e_request_latency_seconds_count{{{f}}}[60s]))",
    ]
    assert metric.parse([0.5, 0.4, 0.9, 1.5, 12.0]) == HistogramResult(avg=0.5, median=0.4, p90=0.9, p99=1.5, per_second=12.0)


def test_duration_renders_as_whole_seconds() -> None:
    assert GaugeMetric("g").get_queries(90.4, "")[0] == "avg_over_time(g{}[90s])"
