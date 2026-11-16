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
"""Unit tests for results.observability in the BR0.2 partial (#799, rows 1 and 3)."""

from typing import Dict, List, Optional

from inference_perf.apis import InferenceInfo, RequestLifecycleMetric
from inference_perf.client.modelserver.metrics import CounterResult, GaugeResult
from inference_perf.client.server_metrics import ModelServerMetrics
from inference_perf.payloads import RequestMetrics, Text
from inference_perf.reportgen.br.v0_2 import build_partial_report
from inference_perf.reportgen.br.v0_2.schema import BenchmarkReportV021


def _stage_summary() -> ModelServerMetrics:
    return ModelServerMetrics(
        kv_cache_usage=GaugeResult(avg=0.42, median=0.4, p90=0.7, p99=0.9),
        num_requests_running=GaugeResult(avg=6.5, median=6.0, p90=9.0, p99=12.0),
        queue_length=GaugeResult(avg=1.5, median=1.0, p90=4.0, p99=7.0),
        num_preemptions_total=CounterResult(total=3.0, avg=0.05, per_second=0.05),
        prefix_cache_hits=CounterResult(total=250.0),
        prefix_cache_queries=CounterResult(total=1000.0),
    )


def _observability(server_metrics: Optional[Dict[str, ModelServerMetrics]]) -> Optional[Dict[str, object]]:
    partial = build_partial_report([], tokenizer=None, run_uid="uid", server_metrics=server_metrics)
    observability: Optional[Dict[str, object]] = partial["results"].get("observability")
    return observability


def _components(observability: Dict[str, object]) -> List[Dict[str, object]]:
    components = observability["components"]
    assert isinstance(components, list)
    return components


def _aggregate(summary: ModelServerMetrics) -> Dict[str, object]:
    observability = _observability({"vllm": summary})
    assert observability is not None
    [component] = _components(observability)
    aggregate = component["aggregate"]
    assert isinstance(aggregate, dict)
    return aggregate


def test_component_label_is_the_server_type() -> None:
    observability = _observability({"vllm": _stage_summary()})

    assert observability is not None
    assert [c["component_label"] for c in _components(observability)] == ["vllm"]


def test_kv_cache_usage_is_a_fraction_with_percentiles() -> None:
    assert _aggregate(_stage_summary())["kv_cache_usage"] == {
        "units": "fraction",
        "mean": 0.42,
        "p50": 0.4,
        "p90": 0.7,
        "p99": 0.9,
    }


def test_running_requests_carry_mean_and_percentiles() -> None:
    assert _aggregate(_stage_summary())["running_requests"] == {
        "units": "count",
        "mean": 6.5,
        "p50": 6.0,
        "p90": 9.0,
        "p99": 12.0,
    }


def test_waiting_requests_come_from_the_queue_length() -> None:
    assert _aggregate(_stage_summary())["waiting_requests"] == {
        "units": "count",
        "mean": 1.5,
        "p50": 1.0,
        "p90": 4.0,
        "p99": 7.0,
    }


def test_preemptions_are_the_stage_total_in_mean() -> None:
    assert _aggregate(_stage_summary())["preemptions"] == {"units": "count", "mean": 3.0}


def test_cache_hit_rate_is_a_percent_of_the_stage_queries() -> None:
    assert _aggregate(_stage_summary())["cache_hit_rate"] == {"units": "percent", "mean": 25.0}


def test_cache_hit_rate_is_left_out_when_nothing_queried_the_cache() -> None:
    summary = _stage_summary()
    summary.prefix_cache_hits = CounterResult(total=0.0)
    summary.prefix_cache_queries = CounterResult(total=0.0)

    aggregate = _aggregate(summary)

    assert "cache_hit_rate" not in aggregate
    assert "kv_cache_usage" in aggregate


def test_metrics_the_server_does_not_declare_are_left_out() -> None:
    summary = ModelServerMetrics(queue_length=GaugeResult(avg=2.0, median=2.0, p90=3.0, p99=4.0))

    assert set(_aggregate(summary)) == {"waiting_requests"}


def test_an_undeclared_queue_length_is_left_out() -> None:
    summary = ModelServerMetrics(kv_cache_usage=GaugeResult(avg=0.1, median=0.1, p90=0.2, p99=0.3))

    assert set(_aggregate(summary)) == {"kv_cache_usage"}


def test_cache_hit_rate_needs_the_hits_declared() -> None:
    summary = ModelServerMetrics(
        queue_length=GaugeResult(avg=1.0),
        prefix_cache_queries=CounterResult(total=1000.0),
    )

    assert "cache_hit_rate" not in _aggregate(summary)


def test_failed_queries_are_left_out() -> None:
    summary = _stage_summary()
    summary.kv_cache_usage = GaugeResult(avg=None, median=None, p90=None, p99=None)
    summary.num_preemptions_total = CounterResult(total=None, avg=None, per_second=None)
    summary.prefix_cache_hits = CounterResult(total=None, avg=None, per_second=None)

    assert set(_aggregate(summary)) == {"running_requests", "waiting_requests"}


def test_cache_hit_rate_is_left_out_when_the_queries_failed() -> None:
    summary = _stage_summary()
    summary.prefix_cache_queries = CounterResult(total=None, avg=None, per_second=None)

    assert "cache_hit_rate" not in _aggregate(summary)


def test_a_gauge_keeps_its_mean_when_the_quantiles_failed() -> None:
    summary = _stage_summary()
    summary.num_requests_running = GaugeResult(avg=6.5, median=None, p90=None, p99=None)

    assert _aggregate(summary)["running_requests"] == {"units": "count", "mean": 6.5}


def test_no_observability_when_nothing_was_measured() -> None:
    summary = ModelServerMetrics(queue_length=GaugeResult(avg=None, median=None, p90=None, p99=None))

    assert _observability({"vllm": summary}) is None


def test_no_observability_without_server_metrics() -> None:
    assert _observability(None) is None


def test_partial_with_observability_is_valid_br() -> None:
    partial = build_partial_report([], tokenizer=None, run_uid="uid", server_metrics={"vllm": _stage_summary()})

    BenchmarkReportV021.model_validate(partial)


# ---------------------------------------------------------------------------
# results.observability.drop_rate (#799, row 3)
# ---------------------------------------------------------------------------


def _request(stage_id: int = 0) -> RequestLifecycleMetric:
    return RequestLifecycleMetric(
        stage_id=stage_id,
        scheduled_time=1.0,
        start_time=1.0,
        end_time=2.0,
        request_data="{}",
        response_data="ok",
        info=InferenceInfo(request_metrics=RequestMetrics(text=Text(input_tokens=10))),
        error=None,
    )


def _drop_observability(recorded: int, dropped: Optional[int]) -> object:
    partial = build_partial_report(
        [_request() for _ in range(recorded)], tokenizer=None, run_uid="uid", dropped_requests=dropped
    )
    BenchmarkReportV021.model_validate(partial)
    observability: object = partial["results"].get("observability")
    return observability


def test_drop_rate_is_the_percent_of_generated_requests_never_dispatched() -> None:
    assert _drop_observability(recorded=3, dropped=1) == {"drop_rate": {"units": "percent", "mean": 25.0}}


def test_drop_rate_is_zero_when_nothing_was_dropped() -> None:
    assert _drop_observability(recorded=4, dropped=0) == {"drop_rate": {"units": "percent", "mean": 0.0}}


def test_drop_rate_is_absent_without_a_count() -> None:
    assert _drop_observability(recorded=4, dropped=None) is None


def test_drop_rate_is_absent_when_the_stage_generated_nothing() -> None:
    assert _drop_observability(recorded=0, dropped=0) is None


def test_drop_rate_sits_beside_the_components() -> None:
    partial = build_partial_report(
        [_request()], tokenizer=None, run_uid="uid", server_metrics={"vllm": _stage_summary()}, dropped_requests=1
    )

    observability = partial["results"]["observability"]
    assert [c["component_label"] for c in observability["components"]] == ["vllm"]
    assert observability["drop_rate"] == {"units": "percent", "mean": 50.0}
