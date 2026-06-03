from typing import List

from inference_perf.client.modelserver.metrics import (
    CounterMetric,
    CounterResult,
    GaugeMetric,
    GaugeResult,
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
