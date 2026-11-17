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
from typing import Any, List, Optional

import pytest

from inference_perf.client.modelserver.metrics.base import Metric
from inference_perf.client.modelserver.metrics import (
    CounterMetric,
    CounterResult,
    GaugeMetric,
    GaugeResult,
    HistogramMetric,
    HistogramResult,
)


def test_gauge_result_as_summary() -> None:
    """as_summary projects avg -> mean and keeps the four percentile keys."""
    summary = GaugeResult(avg=1.0, median=2.0, p90=3.0, p99=4.0).as_summary()

    assert summary == {"mean": 1.0, "median": 2.0, "p90": 3.0, "p99": 4.0}


def test_histogram_result_as_summary_drops_per_second() -> None:
    """HistogramResult inherits as_summary and is narrowed to the gauge keys.

    The extra per_second field is not part of the per-metric report summary.
    """
    summary = HistogramResult(avg=1.0, median=2.0, p90=3.0, p99=4.0, per_second=5.0).as_summary()

    assert summary == {"mean": 1.0, "median": 2.0, "p90": 3.0, "p99": 4.0}
    assert "per_second" not in summary


def test_metric_collect_runs_queries_and_parses() -> None:
    """collect() executes each of the metric's queries in order and parses the results."""
    metric = GaugeMetric(metric_name="vllm:kv_cache_usage_perc")
    seen_queries: List[str] = []

    def execute(query: str) -> float:
        seen_queries.append(query)
        return float(len(seen_queries))  # 1.0, 2.0, 3.0, 4.0 in query order

    result = metric.collect(execute, duration=30, filters="")

    assert seen_queries == metric.get_queries(30, "")
    assert isinstance(result, GaugeResult)
    assert (result.avg, result.median, result.p90, result.p99) == (1.0, 2.0, 3.0, 4.0)


def test_counter_and_histogram_expose_avg_and_per_second() -> None:
    """Both feed prompt_tokens/output_tokens, so both must expose the read fields."""
    for result_type in (CounterResult, HistogramResult):
        fields = result_type.model_fields
        assert "avg" in fields
        assert "per_second" in fields


def test_counter_metric_collects_total_avg_and_per_second() -> None:
    """CounterMetric -> CounterResult: total is the window increase, avg the averaged rate,
    per_second the summed rate. avg uses avg_over_time(rate(...)) (the pre-refactor counter "mean").
    """
    metric = CounterMetric(metric_name="vllm:prompt_tokens")
    queries = metric.get_queries(30, "")

    assert queries == [
        "sum(increase(vllm:prompt_tokens_total{}[30s]) or increase(vllm:prompt_tokens{}[30s]))",
        "avg_over_time((rate(vllm:prompt_tokens_total{}[30s]) or rate(vllm:prompt_tokens{}[30s]))[30s:30s])",
        "sum(rate(vllm:prompt_tokens_total{}[30s]) or rate(vllm:prompt_tokens{}[30s]))",
    ]

    result = metric.collect(lambda q: float(queries.index(q) + 1), duration=30, filters="")

    assert isinstance(result, CounterResult)
    assert (result.total, result.avg, result.per_second) == (1.0, 2.0, 3.0)


def test_counter_metric_spans_both_total_and_bare_names() -> None:
    """A counter is stored as `name_total` (modern prometheus_client exposition) or as the bare
    family name (older exporters), so queries match both exact forms with `or` - never a
    `{__name__=~...}` regex selector, which Google Managed Prometheus rejects (#567). A name
    declared with the `_total` suffix must not get a second one."""
    for declared in ("vllm:request_success", "vllm:request_success_total"):
        queries = CounterMetric(metric_name=declared).get_queries(30, "model_name='m'")
        assert queries[0] == (
            "sum(increase(vllm:request_success_total{model_name='m'}[30s])"
            " or increase(vllm:request_success{model_name='m'}[30s]))"
        )


def test_metric_collect_returns_none_when_a_query_has_no_result() -> None:
    """A failed query passes through parse as a missing value (#822), not a parsed zero."""
    metric = GaugeMetric(metric_name="vllm:kv_cache_usage_perc")

    def execute(query: str) -> Optional[float]:
        return None if "quantile_over_time(0.5" in query else 1.0

    assert metric.collect(execute, duration=30, filters="") == GaugeResult(avg=1.0, median=None, p90=1.0, p99=1.0)


def test_metric_result_is_missing_only_when_all_values_are_missing() -> None:
    """A partially collected metric still carries data; only an all-None result counts as missing."""
    assert GaugeResult().is_missing() is False
    assert GaugeResult(avg=None, median=None, p90=None, p99=None).is_missing() is True
    assert GaugeResult(avg=1.0, median=None, p90=1.0, p99=1.0).is_missing() is False
    assert CounterResult(total=None, avg=None, per_second=None).is_missing() is True


def test_metric_types_reject_name_selectors() -> None:
    """`{__name__=~...}` selector names build queries GMP rejects (regex on `__name__`), and
    gauges/histograms would also wrap or suffix the braces (`{...}{filters}`, `{...}_sum`) into
    invalid PromQL that fails silently at query time, so every type refuses the name up front."""
    for metric_type in (CounterMetric, GaugeMetric, HistogramMetric):
        with pytest.raises(ValueError, match="selector"):
            metric_type(metric_name='{__name__=~"vllm:foo(_total)?"}')


# Counter, gauge and histogram each list the series names they query, and each name must appear
# in the query text followed by `{`. Bare substring matching would let "vllm:prompt_tokens" pass
# inside "vllm:prompt_tokens_total" even if the bare leg were dropped.
def test_candidate_names_are_the_series_each_metric_actually_queries() -> None:
    metrics: List[Metric[Any]] = [
        CounterMetric("vllm:prompt_tokens"),
        GaugeMetric("vllm:num_requests_waiting"),
        HistogramMetric("vllm:e2e_request_latency_seconds"),
    ]
    for metric in metrics:
        queries = " ".join(metric.get_queries(60.0, "model_name='m'"))
        advertised = {name for group in metric.candidate_names() for name in group}
        assert advertised, f"{metric.metric_name} advertises no candidate names"
        unqueried = sorted(name for name in advertised if f"{name}{{" not in queries)
        assert not unqueried, f"{metric.metric_name} advertises {unqueried} but never queries them"


# CounterMetric("vllm:prompt_tokens") and CounterMetric("vllm:prompt_tokens_total") both return
# ({"vllm:prompt_tokens_total"}, {"vllm:prompt_tokens"}): two one-name groups, either one resolves it.
def test_counter_candidate_names_span_both_name_forms() -> None:
    expected = (frozenset({"vllm:prompt_tokens_total"}), frozenset({"vllm:prompt_tokens"}))
    for declared in ("vllm:prompt_tokens", "vllm:prompt_tokens_total"):
        assert CounterMetric(declared).candidate_names() == expected


# CounterMetric("sglang:e2e_request_latency_seconds_count") returns only that exact name, and the
# same holds for _sum and _bucket. Histogram series never carry _total, so no _total leg is added.
@pytest.mark.parametrize("suffix", ["_count", "_sum", "_bucket"])
def test_counter_candidate_names_over_a_histogram_series_are_the_single_exact_name(suffix: str) -> None:
    name = f"sglang:e2e_request_latency_seconds{suffix}"
    assert CounterMetric(name).candidate_names() == (frozenset({name}),)


# HistogramMetric("vllm:lat") returns one group {"vllm:lat_bucket", "vllm:lat_count", "vllm:lat_sum"}.
# A family exposing _sum and _count without _bucket does not satisfy it.
def test_histogram_candidate_names_require_all_three_series_together() -> None:
    assert HistogramMetric("vllm:lat").candidate_names() == (frozenset({"vllm:lat_bucket", "vllm:lat_count", "vllm:lat_sum"}),)


# GaugeMetric("vllm:kv_cache_usage_perc") returns ({"vllm:kv_cache_usage_perc"},) and nothing else.
def test_gauge_candidate_names_are_the_bare_name() -> None:
    assert GaugeMetric("vllm:kv_cache_usage_perc").candidate_names() == (frozenset({"vllm:kv_cache_usage_perc"}),)
