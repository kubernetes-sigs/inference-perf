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
"""Regression tests for request completion windows and out-of-order delivery (#928)."""

from datetime import datetime, timedelta
from queue import Queue
from unittest.mock import patch

import pytest
from pydantic import BaseModel

import inference_perf.circuit_breaker as cb_module
from inference_perf.apis import InferenceInfo, RequestLifecycleMetric
from inference_perf.circuit_breaker import feed_breakers
from inference_perf.circuit_breaker.simple_breaker import SimpleCircuitBreaker
from inference_perf.circuit_breaker.triggers import HitSample
from inference_perf.circuit_breaker.triggers.rate_over_window import RateOverWindow
from inference_perf.config.circuit_breaker import CircuitBreakerConfig, MetricsSpec, TriggerRateOverWindow
from inference_perf.metrics.request_collector import RequestMetricCollector
from inference_perf.metrics.request_collector.local import LocalRequestMetricCollector
from inference_perf.metrics.request_collector.multiprocess import MultiprocessRequestMetricCollector
from inference_perf.payloads import RequestMetrics, Text

BASE = datetime(2026, 1, 1)


class _GenericMetric(BaseModel):
    stage_id: int


class _UnrelatedEndTimeMetric(_GenericMetric):
    end_time: float


# A request completes at end_time on the perf_counter timeline; stage 1 is a hit,
# stage 0 is a miss, and the request started one second before completion.
def _metric(end_time: float, hit: int) -> RequestLifecycleMetric:
    return RequestLifecycleMetric(
        stage_id=hit,
        scheduled_time=end_time - 1,
        start_time=end_time - 1,
        end_time=end_time,
        request_data="request",
        info=InferenceInfo(request_metrics=RequestMetrics(text=Text(input_tokens=1))),
        error=None,
    )


# Requests complete at 10/11 (misses) and 30/31 (hits). A 10s window with min=2
# must open only at 31, whether notifications happen at completion or at 100.
@pytest.mark.parametrize("multiprocess", [False, True], ids=["local", "multiprocess"])
@pytest.mark.parametrize("delayed", [False, True], ids=["at-completion", "delayed-drain"])
async def test_collector_window_uses_request_completion_times(
    monkeypatch: pytest.MonkeyPatch, multiprocess: bool, delayed: bool
) -> None:
    breaker = SimpleCircuitBreaker(
        CircuitBreakerConfig(
            name="request-timing",
            metrics=MetricsSpec(matches=["stage_id >= `0`"], rules=["stage_id == `1`"]),
            triggers=[TriggerRateOverWindow(type="rate_over_window", window_sec=10, threshold=0.75, min_samples=2)],
        )
    )
    monkeypatch.setattr(cb_module, "_initialized_circuit_breakers", {breaker.name: breaker})
    collector: RequestMetricCollector
    if multiprocess:
        # Keep record_metric(), executor draining, and observer delivery real, but
        # remove the multiprocessing queue's asynchronous feeder from this timeline.
        queue: Queue[RequestLifecycleMetric | None] = Queue()
        with patch("inference_perf.metrics.request_collector.multiprocess.MP_CONTEXT.JoinableQueue", return_value=queue):
            collector = MultiprocessRequestMetricCollector()
    else:
        collector = LocalRequestMetricCollector()
    collector.add_observer(feed_breakers)
    opened: list[bool] = []
    collector.add_observer(lambda _: opened.append(breaker.is_open()))
    metrics = [_metric(10, 0), _metric(11, 0), _metric(30, 1), _metric(31, 1)]
    processing_times = [100, 100, 100, 100] if delayed else [10, 11, 30, 31]

    # create=True also allows the production fix to stop importing datetime.
    with patch("inference_perf.circuit_breaker.simple_breaker.datetime", wraps=datetime, create=True) as clock:
        clock.now.side_effect = [BASE + timedelta(seconds=t) for t in processing_times]
        for metric in metrics:
            collector.record_metric(metric)
        if isinstance(collector, MultiprocessRequestMetricCollector):
            collector.queue.put(None)
            collected = await collector.collect_metrics()
        else:
            collected = collector.get_metrics()

    assert collected == metrics
    assert opened == [False, False, False, True]


# Arrival 20(miss), 12(miss), 15(hit) must be ordered 12/15/20. At 23(hit),
# evict 12 and fire with 2/3 hits >= 0.6; retaining 12 would leave only 2/4.
def test_window_orders_active_samples_by_completion_time() -> None:
    trigger = RateOverWindow(window_sec=10, threshold=0.6, min_samples=3)
    samples = [HitSample(BASE + timedelta(seconds=t), hit) for t, hit in [(20, 0), (12, 0), (15, 1)]]
    for sample in samples:
        trigger.update(sample)

    assert trigger.fired() is False
    active_before_eviction = list(trigger.buf)
    latest = HitSample(BASE + timedelta(seconds=23), 1)
    trigger.update(latest)

    assert trigger.fired() is True
    assert active_before_eviction == [samples[1], samples[2], samples[0]]
    assert list(trigger.buf) == [samples[2], samples[0], latest]


# After 20(miss), a late 5(hit) is outside [10,20]. It must neither satisfy
# min_samples=2 nor turn the rate into 1/2 and incorrectly latch the trigger.
def test_expired_late_hit_does_not_fire_window() -> None:
    trigger = RateOverWindow(window_sec=10, threshold=0.5, min_samples=2)
    newer = HitSample(BASE + timedelta(seconds=20), 0)
    trigger.update(newer)
    trigger.update(HitSample(BASE + timedelta(seconds=5), 1))

    assert trigger.fired() is False
    assert list(trigger.buf) == [newer]


# Arrival 20(miss), 5(miss), 21(hit): discard 5 immediately against latest=20,
# then fire at 21 with 1/2 hits. An expired 5 trapped behind 20 would yield 1/3.
def test_expired_late_miss_does_not_hide_later_threshold() -> None:
    trigger = RateOverWindow(window_sec=10, threshold=0.5, min_samples=2)
    newer = HitSample(BASE + timedelta(seconds=20), 0)
    trigger.update(newer)
    trigger.update(HitSample(BASE + timedelta(seconds=5), 0))
    active_after_late = list(trigger.buf)
    assert trigger.fired() is False

    latest = HitSample(BASE + timedelta(seconds=21), 1)
    trigger.update(latest)

    assert trigger.fired() is True
    assert active_after_late == [newer]
    assert list(trigger.buf) == [newer, latest]


# At latest=20, a miss exactly at cutoff=10 remains: the two misses and one hit
# must keep a min=2, threshold=0.5 window closed (dropping 10 would fire it).
def test_window_retains_sample_exactly_at_cutoff() -> None:
    trigger = RateOverWindow(window_sec=10, threshold=0.5, min_samples=2)
    for t, hit in [(10, 0), (19, 0), (20, 1)]:
        trigger.update(HitSample(BASE + timedelta(seconds=t), hit))

    assert trigger.fired() is False
    assert len(trigger.buf) == 3


# Fire at 20/21, then a miss at 40 must not unlatch. After reset, an earlier
# timeline 1/2 must fire afresh; no previous latest-time state may evict it.
def test_window_reset_starts_fresh_completion_timeline() -> None:
    trigger = RateOverWindow(window_sec=10, threshold=1, min_samples=2)
    trigger.update(HitSample(BASE + timedelta(seconds=20), 1))
    trigger.update(HitSample(BASE + timedelta(seconds=21), 1))
    assert trigger.fired() is True
    trigger.update(HitSample(BASE + timedelta(seconds=40), 0))
    assert trigger.fired() is True

    trigger.reset()
    assert trigger.fired() is False
    assert not trigger.buf
    trigger.update(HitSample(BASE + timedelta(seconds=1), 1))
    assert trigger.fired() is False
    trigger.update(HitSample(BASE + timedelta(seconds=2), 1))
    assert trigger.fired() is True
    assert len(trigger.buf) == 2


# Requests complete at 100(miss)/120(hit); generic samples arrive at 101(miss)/121(hit).
# Both model types must share one clock domain and open at 121, ignoring an unrelated epoch end_time.
@pytest.mark.parametrize("has_end_time", [False, True], ids=["without-timing", "unrelated-end-time"])
def test_generic_models_share_request_clock_domain(has_end_time: bool) -> None:
    breaker = SimpleCircuitBreaker(
        CircuitBreakerConfig(
            name="generic-timing",
            metrics=MetricsSpec(matches=["stage_id >= `0`"], rules=["stage_id == `1`"]),
            triggers=[TriggerRateOverWindow(type="rate_over_window", window_sec=10, threshold=0.75, min_samples=2)],
        )
    )
    generic = [
        _UnrelatedEndTimeMetric(stage_id=hit, end_time=1_700_000_000) if has_end_time else _GenericMetric(stage_id=hit)
        for hit in [0, 1]
    ]
    opened: list[bool] = []
    with patch("inference_perf.circuit_breaker.simple_breaker.time.perf_counter", side_effect=[101.0, 121.0]) as clock:
        for metric in [_metric(100, 0), generic[0], _metric(120, 1), generic[1]]:
            breaker.feed(metric)
            opened.append(breaker.is_open())

    assert opened == [False, False, False, True]
    assert clock.call_count == 2


# Map zero, negative, fractional, and long-uptime request seconds without reading
# the processing clock. Fractional seconds round to the existing datetime microsecond resolution.
@pytest.mark.parametrize(
    ("end_time", "expected_microseconds"),
    [
        (0.0, 0),
        (-1.0, -1_000_000),
        (1_000_000.1234564, 1_000_000_123456),
        (1_000_000.1234566, 1_000_000_123457),
        (1_000_000_000.125, 1_000_000_000_125000),
    ],
)
def test_request_time_mapping_preserves_microsecond_precision(end_time: float, expected_microseconds: int) -> None:
    breaker = SimpleCircuitBreaker(
        CircuitBreakerConfig(
            name="precision",
            metrics=MetricsSpec(matches=["stage_id == `1`"]),
            triggers=[TriggerRateOverWindow(type="rate_over_window", window_sec=10, threshold=1, min_samples=2)],
        )
    )
    with patch(
        "inference_perf.circuit_breaker.simple_breaker.time.perf_counter",
        side_effect=AssertionError("Request completion must not read the processing clock"),
    ):
        breaker.feed(_metric(end_time, 1))

    trigger = breaker._triggers[0]
    assert isinstance(trigger, RateOverWindow)
    assert len(trigger.buf) == 1
    assert trigger.buf[0].ts == datetime(2000, 1, 1) + timedelta(microseconds=expected_microseconds)
    assert trigger.buf[0].ts.tzinfo is None


# Arrival 20(miss), 10(miss), 10(hit), 20(hit) retains both duplicate pairs in
# arrival order within each timestamp, reaching min=4 and firing at exactly 2/4.
def test_window_retains_equal_timestamps_on_append_and_late_insertion() -> None:
    trigger = RateOverWindow(window_sec=10, threshold=0.5, min_samples=4)
    samples = [HitSample(BASE + timedelta(seconds=t), hit) for t, hit in [(20, 0), (10, 0), (10, 1), (20, 1)]]
    for sample in samples[:-1]:
        trigger.update(sample)
        assert trigger.fired() is False

    trigger.update(samples[-1])

    assert trigger.fired() is True
    assert list(trigger.buf) == [samples[1], samples[2], samples[0], samples[3]]


# Into 10/12/12/16/20, insert a hit at 19 (near tail), 9 (before all), or 12
# (after both equals). Each preserves all samples and fires at exactly 3/6.
@pytest.mark.parametrize(("seconds", "position"), [(19, 4), (9, 0), (12, 3)])
def test_window_late_insertion_positions(seconds: int, position: int) -> None:
    trigger = RateOverWindow(window_sec=20, threshold=0.5, min_samples=6)
    samples = [HitSample(BASE + timedelta(seconds=t), hit) for t, hit in [(10, 0), (12, 1), (12, 0), (16, 1), (20, 0)]]
    for sample in samples:
        trigger.update(sample)
        assert trigger.fired() is False

    late = HitSample(BASE + timedelta(seconds=seconds), 1)
    trigger.update(late)
    expected = samples.copy()
    expected.insert(position, late)

    assert trigger.fired() is True
    assert list(trigger.buf) == expected
