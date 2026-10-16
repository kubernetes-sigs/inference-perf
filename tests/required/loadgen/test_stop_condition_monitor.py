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
"""``StopConditionMonitor`` ends a stage on the results it observes.

Counters are exact and end the stage on the result that flipped them. A windowed
aggregate has no value until its window is full, and ``hold`` asks for the
condition on several consecutive results. The per-result values it reads are
pinned to the report's definitions.
"""

import pytest

from inference_perf.apis import (
    ErrorResponseInfo,
    InferenceInfo,
    RequestLifecycleMetric,
    ResponseMetrics,
    StreamedResponseMetrics,
    UnaryResponseMetrics,
)
from inference_perf.loadgen.stop_condition import StopConditionMonitor, per_result_values
from inference_perf.payloads import RequestMetrics, Text
from inference_perf.reportgen.base import compute_request_latency_metrics
from inference_perf.utils.numeric.expression import Predicate


# One result for `stage_id` that started at 0 and ended at `latency`; failed when `error`; streamed with
# `output_tokens` evenly spaced token times from `ttft` to `latency` when `output_tokens` is set.
def _result(
    latency: float = 0.1, *, stage_id: int = 0, error: bool = False, output_tokens: int = 0, ttft: float = 0.05
) -> RequestLifecycleMetric:
    response: ResponseMetrics
    if output_tokens:
        step = (latency - ttft) / max(output_tokens - 1, 1)
        response = StreamedResponseMetrics(
            output_tokens=output_tokens, output_token_times=[ttft + i * step for i in range(output_tokens)]
        )
    else:
        response = UnaryResponseMetrics(output_tokens=0)
    return RequestLifecycleMetric(
        stage_id=stage_id,
        scheduled_time=0.0,
        start_time=0.0,
        end_time=latency,
        request_data="x",
        info=InferenceInfo(request_metrics=RequestMetrics(text=Text(input_tokens=1)), response_metrics=response),
        error=ErrorResponseInfo(error_type="TimeoutError", error_msg="timed out") if error else None,
    )


# A monitor for '(t >= 300) | (errors >= 3)' on stage 0 started at clock 100, with window 100 and hold `hold`.
def _monitor(condition: str = "(t >= 300) | (errors >= 3)", *, window: int = 100, hold: int = 1) -> StopConditionMonitor:
    monitor = StopConditionMonitor(0, Predicate(condition), window=window, hold=hold)
    monitor.start(100.0)
    return monitor


# Three failed results observed at clock 101, 102, 103: met_at is None after two and 3.0 (stage seconds) on the
# third, the result that made errors reach 3; the values at that moment and the totals are recorded.
def test_counter_ends_on_the_flipping_result() -> None:
    monitor = _monitor()
    monitor.observe(_result(error=True), now=101.0)
    monitor.observe(_result(error=True), now=102.0)
    assert monitor.met_at is None
    monitor.observe(_result(error=True), now=103.0)
    assert monitor.met_at == 3.0
    assert monitor.values_when_met == {"errors": 3.0}
    assert (monitor.completed, monitor.errors) == (3, 3)


# hold=3: errors reaches 3 on the third failure, but the condition must hold on three consecutive results, so
# met_at stays None through the fourth and is 5.0 on the fifth.
def test_hold_requires_consecutive_results() -> None:
    monitor = _monitor(hold=3)
    for second in (101.0, 102.0, 103.0, 104.0):
        monitor.observe(_result(error=True), now=second)
        assert monitor.met_at is None, second
    monitor.observe(_result(error=True), now=105.0)
    assert monitor.met_at == 5.0


# '(t >= 300) | (latency_p50 > 0.05)' with window 4: 0.1s results give the aggregate no value (values() omits it)
# until the fourth, which fills the window and ends the stage at 4.0 with latency_p50 0.1.
def test_aggregate_has_no_value_until_window_full() -> None:
    monitor = _monitor("(t >= 300) | (latency_p50 > 0.05)", window=4)
    for second in (101.0, 102.0, 103.0):
        monitor.observe(_result(0.1), now=second)
        assert monitor.met_at is None and monitor.values() == {}
    monitor.observe(_result(0.1), now=104.0)
    assert monitor.met_at == 4.0
    assert monitor.values_when_met == {"latency_p50": pytest.approx(0.1)}


# A window of 3 slides: under '(t >= 300) | (latency_mean < 0.1)' latencies 1, 1, 1, 0.01, 0.01, 0.01 end the stage
# on the sixth result, when the window holds only the three fast ones (mean 0.01); the fifth still holds a 1 (mean 0.34).
def test_window_slides() -> None:
    monitor = _monitor("(t >= 300) | (latency_mean < 0.1)", window=3)
    for i, latency in enumerate((1.0, 1.0, 1.0, 0.01, 0.01), start=1):
        monitor.observe(_result(latency), now=100.0 + i)
        assert monitor.met_at is None, i
    monitor.observe(_result(0.01), now=106.0)
    assert monitor.met_at == 6.0
    assert monitor.values_when_met == {"latency_mean": pytest.approx(0.01)}


# A failed result carries no latency: under '(t >= 300) | (latency_p50 > 0)' with window 1 a failure leaves the
# window empty (errors 1, no value), and the following 0.1s success ends the stage at 2.0.
def test_failed_results_have_no_latency() -> None:
    monitor = _monitor("(t >= 300) | (latency_p50 > 0)", window=1)
    monitor.observe(_result(error=True), now=101.0)
    assert monitor.met_at is None
    assert monitor.values() == {}
    assert monitor.errors == 1
    monitor.observe(_result(0.1), now=102.0)
    assert monitor.met_at == 2.0


# '((t >= 60) & (errors >= 1)) | (t >= 600)' cannot hold before 60: a failure at t=10 is counted (errors 1) but
# does not end the stage, and a success at t=61 then does, since errors >= 1 already holds.
def test_results_before_earliest_cannot_end_the_stage() -> None:
    monitor = _monitor("((t >= 60) & (errors >= 1)) | (t >= 600)")
    monitor.observe(_result(error=True), now=110.0)
    assert monitor.met_at is None
    assert monitor.errors == 1
    monitor.observe(_result(0.1), now=161.0)
    assert monitor.met_at == 61.0


# The hold counter resets on a result where the condition does not hold: '(t >= 300) | (latency_mean > 0.5)' with
# window 1 and hold 2 on latencies 1, 0.1, 1, 1 ends on the fourth result, not the third.
def test_hold_resets_when_condition_lapses() -> None:
    monitor = _monitor("(t >= 300) | (latency_mean > 0.5)", window=1, hold=2)
    for i, latency in enumerate((1.0, 0.1, 1.0), start=1):
        monitor.observe(_result(latency), now=100.0 + i)
        assert monitor.met_at is None, i
    monitor.observe(_result(1.0), now=104.0)
    assert monitor.met_at == 4.0


# A stage-1 failure observed by stage 0's monitor changes nothing: completed and errors stay 0.
def test_other_stage_results_are_ignored() -> None:
    monitor = _monitor()
    monitor.observe(_result(stage_id=1, error=True), now=101.0)
    assert (monitor.completed, monitor.errors, monitor.met_at) == (0, 0, None)


# Results are ignored before start() and after the condition has met: an unstarted monitor counts nothing, and
# once met at 3.0 a fourth failure leaves met_at 3.0 and completed 3.
def test_not_started_or_already_met() -> None:
    unstarted = StopConditionMonitor(0, Predicate("(t >= 300) | (errors >= 3)"), window=100, hold=1)
    unstarted.observe(_result(error=True), now=101.0)
    assert (unstarted.completed, unstarted.met_at) == (0, None)
    monitor = _monitor()
    for second in (101.0, 102.0, 103.0, 104.0):
        monitor.observe(_result(error=True), now=second)
    assert (monitor.met_at, monitor.completed) == (3.0, 3)


# Only the metrics the condition reads are kept: '(t >= 300) | (errors >= 3)' keeps no latency samples, and
# '(t >= 300) | (ttft_p90 > 1)' keeps ttft samples only.
def test_only_named_metrics_are_sampled() -> None:
    assert _monitor()._samples == {}
    assert list(_monitor("(t >= 300) | (ttft_p90 > 1)")._samples) == ["ttft"]


# per_result_values matches the report's compute_request_latency_metrics for a streamed success of 5 tokens ending
# at 0.3s with first token at 0.05s: latency 0.3, ttft 0.05, tpot (0.3 - 0.05) / 4; a unary success gives latency
# alone and a failure nothing.
def test_per_result_values_match_the_report() -> None:
    streamed = _result(0.3, output_tokens=5, ttft=0.05)
    report = compute_request_latency_metrics(streamed)
    values = per_result_values(streamed)
    assert values == pytest.approx(
        {
            "latency": report["request_latency"],
            "ttft": report["time_to_first_token"],
            "tpot": report["time_per_output_token"],
        }
    )
    assert values["tpot"] == pytest.approx((0.3 - 0.05) / 4)
    assert per_result_values(_result(0.2)) == {"latency": pytest.approx(0.2)}
    assert per_result_values(_result(0.2, error=True)) == {}
