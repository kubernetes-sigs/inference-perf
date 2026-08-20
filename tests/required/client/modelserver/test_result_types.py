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
        "sum(increase(vllm:prompt_tokens{}[30s]))",
        "avg_over_time(rate(vllm:prompt_tokens{}[30s])[30s:30s])",
        "sum(rate(vllm:prompt_tokens{}[30s]))",
    ]

    result = metric.collect(lambda q: float(queries.index(q) + 1), duration=30, filters="")

    assert isinstance(result, CounterResult)
    assert (result.total, result.avg, result.per_second) == (1.0, 2.0, 3.0)


def test_counter_metric_merges_filters_into_name_selector() -> None:
    """A counter whose name is a `{__name__=~...}` selector (e.g. the requests count) merges
    filters inside the braces rather than appending a second `{...}` group."""
    metric = CounterMetric(metric_name='{__name__=~"vllm:request_success(_total)?"}')
    queries = metric.get_queries(30, "model_name='m'")

    assert queries[0] == "sum(increase({__name__=~\"vllm:request_success(_total)?\",model_name='m'}[30s]))"
    assert queries[2] == "sum(rate({__name__=~\"vllm:request_success(_total)?\",model_name='m'}[30s]))"


def test_gauge_and_histogram_reject_name_selectors() -> None:
    """`{__name__=~...}` selector names are counter-only; the other metric types wrap or
    suffix the name (`{...}{filters}`, `{...}_sum`), which builds invalid PromQL that would
    fail silently at query time, so they must refuse the name up front."""
    for metric_type in (GaugeMetric, HistogramMetric):
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
    # The `{__name__=~"X(_total)?"}` counter form is deliberately not in this list:
    # it spells its alternatives as a regex, so "vllm:request_success_total" is
    # never literally in the query text and containment cannot judge it. Its own
    # test below pins it instead.
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


def test_counter_candidate_names_report_a_plain_name_as_selecting_only_itself() -> None:
    # CounterMetric("vllm:prompt_tokens") builds increase(vllm:prompt_tokens{...}),
    # with no _total suffix anywhere, so it selects exactly one series. Reporting
    # the bare name alone is what lets a drift check notice that a server exposing
    # only vllm:prompt_tokens_total leaves this query matching nothing (#669).
    assert CounterMetric("vllm:prompt_tokens").candidate_names() == (frozenset({"vllm:prompt_tokens"}),)


def test_counter_candidate_names_span_both_forms_for_a_selector_name() -> None:
    # A `{__name__=~"X(_total)?"}` declaration selects either exact form, so it
    # reports two single-name groups: satisfying either one resolves the metric.
    names = CounterMetric('{__name__=~"vllm:request_success(_total)?"}').candidate_names()
    assert names == (frozenset({"vllm:request_success_total"}), frozenset({"vllm:request_success"}))


def test_histogram_candidate_names_require_all_three_series_together() -> None:
    # HistogramMetric("vllm:lat") queries _sum, _count and _bucket, so all three
    # go in ONE group: a family exposing _sum and _count but no _bucket is drift,
    # not a metric that half works.
    assert HistogramMetric("vllm:lat").candidate_names() == (frozenset({"vllm:lat_bucket", "vllm:lat_count", "vllm:lat_sum"}),)


def test_gauge_candidate_names_are_the_bare_name() -> None:
    # GaugeMetric("vllm:kv_cache_usage_perc") queries the bare name and nothing else.
    assert GaugeMetric("vllm:kv_cache_usage_perc").candidate_names() == (frozenset({"vllm:kv_cache_usage_perc"}),)
