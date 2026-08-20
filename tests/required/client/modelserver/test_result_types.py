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
from typing import Any, List

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


def test_metric_types_reject_name_selectors() -> None:
    """`{__name__=~...}` selector names build queries GMP rejects (regex on `__name__`), and
    gauges/histograms would also wrap or suffix the braces (`{...}{filters}`, `{...}_sum`) into
    invalid PromQL that fails silently at query time, so every type refuses the name up front."""
    for metric_type in (CounterMetric, GaugeMetric, HistogramMetric):
        with pytest.raises(ValueError, match="selector"):
            metric_type(metric_name='{__name__=~"vllm:foo(_total)?"}')


def test_candidate_names_are_the_series_each_metric_actually_queries() -> None:
    # The anti-drift invariant: whatever candidate_names() reports must literally
    # appear in the metric's own get_queries() output. A gauge "vllm:queue" says
    # [{"vllm:queue"}] and queries avg_over_time(vllm:queue{...}); a histogram
    # "vllm:lat" says [{"vllm:lat_bucket","vllm:lat_count","vllm:lat_sum"}] and
    # queries all three. If someone changes a query builder without changing the
    # names it advertises, this test is what goes red.
    #
    # The counter now belongs here too: it spans both name forms as two exact legs
    # (`increase(X_total{...}) or increase(X{...})`), so both advertised names appear
    # literally in the query text and containment can judge them (#568).
    metrics: List[Metric[Any]] = [
        CounterMetric("vllm:prompt_tokens"),
        GaugeMetric("vllm:num_requests_waiting"),
        HistogramMetric("vllm:e2e_request_latency_seconds"),
    ]
    for metric in metrics:
        queries = " ".join(metric.get_queries(60.0, "model_name='m'"))
        advertised = {name for group in metric.candidate_names() for name in group}
        assert advertised, f"{metric.metric_name} advertises no candidate names"
        unqueried = sorted(name for name in advertised if name not in queries)
        assert not unqueried, f"{metric.metric_name} advertises {unqueried} but never queries them"


def test_counter_candidate_names_span_both_name_forms() -> None:
    # CounterMetric("vllm:prompt_tokens") queries increase(vllm:prompt_tokens_total{...})
    # or increase(vllm:prompt_tokens{...}), so it reports two single-name groups:
    # satisfying either one resolves the metric. Declaring the name with the _total
    # suffix already on it selects the same two series, so it reports the same groups.
    expected = (frozenset({"vllm:prompt_tokens_total"}), frozenset({"vllm:prompt_tokens"}))
    for declared in ("vllm:prompt_tokens", "vllm:prompt_tokens_total"):
        assert CounterMetric(declared).candidate_names() == expected


def test_counter_candidate_names_over_a_histogram_series_are_the_single_exact_name() -> None:
    # A _count/_sum/_bucket series can never carry a _total suffix, so a counter over one
    # queries the single exact leg and must advertise only that name: reporting
    # "sglang:e2e_request_latency_seconds_count_total" would make the drift check demand
    # a series no exposition ever produces.
    metric = CounterMetric("sglang:e2e_request_latency_seconds_count")
    assert metric.candidate_names() == (frozenset({"sglang:e2e_request_latency_seconds_count"}),)


def test_histogram_candidate_names_require_all_three_series_together() -> None:
    # HistogramMetric("vllm:lat") queries _sum, _count and _bucket, so all three
    # go in ONE group: a family exposing _sum and _count but no _bucket is drift,
    # not a metric that half works.
    assert HistogramMetric("vllm:lat").candidate_names() == (frozenset({"vllm:lat_bucket", "vllm:lat_count", "vllm:lat_sum"}),)


def test_gauge_candidate_names_are_the_bare_name() -> None:
    # GaugeMetric("vllm:kv_cache_usage_perc") queries the bare name and nothing else.
    assert GaugeMetric("vllm:kv_cache_usage_perc").candidate_names() == (frozenset({"vllm:kv_cache_usage_perc"}),)
