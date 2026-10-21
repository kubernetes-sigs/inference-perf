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
"""``StopConditionMonitor`` ends a stage on the results it observes and on its once-a-second tick.

Signals are exact and end the stage on the result that flipped them. A windowed
term has no value until its window is complete, and ``hold`` asks for the
condition on several consecutive evaluations. The per-result values it feeds
the terms are pinned to the report's definitions.
"""

from typing import Optional

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
from inference_perf.reportgen.base import compute_request_latency_metrics, summarize_requests
from inference_perf.utils.numeric.expression import Predicate


# One result for `stage_id` scheduled at `scheduled`, started at `start` and ended `latency` later, on the
# monitor's clock; failed when `error`; streamed with `output_tokens` evenly spaced token times from `ttft` after
# start to the end when `output_tokens` is set.
def _result(
    latency: float = 0.1,
    *,
    start: float = 100.0,
    scheduled: Optional[float] = None,
    stage_id: int = 0,
    error: bool = False,
    output_tokens: int = 0,
    ttft: float = 0.05,
    server_output_tokens: Optional[int] = None,
) -> RequestLifecycleMetric:
    response: ResponseMetrics
    if output_tokens:
        step = (latency - ttft) / max(output_tokens - 1, 1)
        response = StreamedResponseMetrics(
            output_tokens=output_tokens, output_token_times=[start + ttft + i * step for i in range(output_tokens)]
        )
    else:
        response = UnaryResponseMetrics(output_tokens=0)
    if server_output_tokens is not None:
        response.server_usage = {"completion_tokens": server_output_tokens}
    return RequestLifecycleMetric(
        stage_id=stage_id,
        scheduled_time=start if scheduled is None else scheduled,
        start_time=start,
        end_time=start + latency,
        request_data="x",
        info=InferenceInfo(request_metrics=RequestMetrics(text=Text(input_tokens=1)), response_metrics=response),
        error=ErrorResponseInfo(error_type="TimeoutError", error_msg="timed out") if error else None,
    )


# A monitor for `condition` on stage 0 started at clock 100, with hold `hold`.
def _monitor(condition: str = "(t >= 300) | (errors >= 3)", *, hold: int = 1) -> StopConditionMonitor:
    monitor = StopConditionMonitor(0, Predicate(condition), hold=hold)
    monitor.start(100.0)
    return monitor


# Three failed results observed at clock 101, 102, 103: met_at is None after two and 3.0 (stage seconds) on the
# third, the result that made errors reach 3; the values at that moment are recorded by name.
def test_signal_ends_on_the_flipping_result() -> None:
    monitor = _monitor()
    monitor.observe(_result(error=True), now=101.0)
    monitor.observe(_result(error=True), now=102.0)
    assert monitor.met_at is None
    monitor.observe(_result(error=True), now=103.0)
    assert monitor.met_at == 3.0
    assert monitor.values_when_met == {"errors": 3.0}


# hold=3: errors reaches 3 on the third failure, but the condition must hold on three consecutive evaluations, so
# met_at stays None through the fourth and is 5.0 on the fifth.
def test_hold_requires_consecutive_evaluations() -> None:
    monitor = _monitor(hold=3)
    for second in (101.0, 102.0, 103.0, 104.0):
        monitor.observe(_result(error=True), now=second)
        assert monitor.met_at is None, second
    monitor.observe(_result(error=True), now=105.0)
    assert monitor.met_at == 5.0


# The hold counter resets on an evaluation where the condition does not hold: '(t >= 300) |
# (avg_over_last(latency, 1) > 0.5)' with hold 2 on latencies 1, 0.1, 1, 1 ends on the fourth result, not the third.
def test_hold_resets_when_condition_lapses() -> None:
    monitor = _monitor("(t >= 300) | (avg_over_last(latency, 1) > 0.5)", hold=2)
    for i, latency in enumerate((1.0, 0.1, 1.0), start=1):
        monitor.observe(_result(latency), now=100.0 + i)
        assert monitor.met_at is None, i
    monitor.observe(_result(1.0), now=104.0)
    assert monitor.met_at == 4.0


# 'quantile_over_last(0.5, latency, 4) > 0.05' on 0.1s results has no value through the third result and ends the
# stage on the fourth, which completes the window; values_when_met names the term as written.
def test_window_term_has_no_value_until_complete() -> None:
    monitor = _monitor("(t >= 300) | (quantile_over_last(0.5, latency, 4) > 0.05)")
    for second in (101.0, 102.0, 103.0):
        monitor.observe(_result(0.1), now=second)
        assert monitor.met_at is None
    monitor.observe(_result(0.1), now=104.0)
    assert monitor.met_at == 4.0
    assert monitor.values_when_met == {"quantile_over_last(0.5, latency, 4)": pytest.approx(0.1)}


# A time range is placed by when each request finished on the stage clock: two 0.5s errors finishing at stage
# seconds 2 and 9 are both inside 'sum_over_time(error[10s]) >= 2' at t=10 (the range lies inside the stage by then).
def test_time_range_uses_finish_time() -> None:
    monitor = _monitor("(t >= 300) | (sum_over_time(error[10s]) >= 2)")
    monitor.observe(_result(0.5, start=101.5, error=True), now=102.0)
    monitor.observe(_result(0.5, start=108.5, error=True), now=109.0)
    assert monitor.met_at is None
    monitor.tick(110.0)
    assert monitor.met_at == 10.0


# '((t >= 60) & (errors >= 1)) | (t >= 600)' cannot hold before 60: a failure at t=10 is counted but does not end
# the stage, and a success at t=61 then does, since errors >= 1 already holds.
def test_results_before_earliest_cannot_end_the_stage() -> None:
    monitor = _monitor("((t >= 60) & (errors >= 1)) | (t >= 600)")
    monitor.observe(_result(error=True), now=110.0)
    assert monitor.met_at is None
    monitor.observe(_result(0.1), now=161.0)
    assert monitor.met_at == 61.0


# A stage-1 failure observed by stage 0's monitor changes nothing: errors stays 0.
def test_other_stage_results_are_ignored() -> None:
    monitor = _monitor()
    monitor.observe(_result(stage_id=1, error=True), now=101.0)
    assert (monitor.state.completed, monitor.state.errors, monitor.met_at) == (0, 0, None)


# Results are ignored before start() and after the condition has held: an unstarted monitor counts nothing, and
# once met at 3.0 a fourth failure leaves met_at 3.0 and completed 3.
def test_not_started_or_already_met() -> None:
    unstarted = StopConditionMonitor(0, Predicate("(t >= 300) | (errors >= 3)"), hold=1)
    unstarted.observe(_result(error=True), now=101.0)
    unstarted.tick(150.0)
    assert (unstarted.state.completed, unstarted.met_at) == (0, None)
    monitor = _monitor()
    for second in (101.0, 102.0, 103.0, 104.0):
        monitor.observe(_result(error=True), now=second)
    monitor.tick(110.0)
    assert (monitor.met_at, monitor.state.completed) == (3.0, 3)


# A stalled server ends the stage on a tick, with no result arriving: '(t >= 300) | ((in_flight > 0) &
# (since_last_completion > 5))' with 2 requests in flight holds at the tick at clock 106 (t=6), not at 105.
def test_tick_ends_a_stall() -> None:
    monitor = _monitor("(t >= 300) | ((in_flight > 0) & (since_last_completion > 5))")
    monitor.tick(105.0, in_flight=2)
    assert monitor.met_at is None
    monitor.tick(106.0, in_flight=2)
    assert monitor.met_at == 6.0
    assert monitor.values_when_met == {"in_flight": 2.0, "since_last_completion": 6.0}


# The tick evaluates only after a second without an evaluation: none in the stage's first second, and none
# within a second of a result. With hold 2 on errors >= 1, a failure at 101 counts once, a tick at 101.5 is
# skipped, and a tick at 102.1 is the second consecutive evaluation, ending the stage at 2.1.
def test_tick_is_once_a_second_without_evaluations() -> None:
    monitor = _monitor("(t >= 300) | (errors >= 1)", hold=2)
    monitor.tick(100.5)
    monitor.observe(_result(error=True), now=101.0)
    monitor.tick(101.5)
    assert monitor.met_at is None
    monitor.tick(102.1)
    assert monitor.met_at == pytest.approx(2.1)


# The first tick comes a second into the stage, even when the condition already holds:
# 'since_last_completion > 0.2' holds at t=0.5 but is first evaluated, and met, at t=1.
def test_first_tick_is_a_second_into_the_stage() -> None:
    monitor = _monitor("(t >= 300) | (since_last_completion > 0.2)")
    monitor.tick(99.5)
    monitor.tick(100.5)
    assert monitor.met_at is None
    monitor.tick(101.0)
    assert monitor.met_at == 1.0


# Dispatch order places results for _over_sent: requests scheduled at 100.0, 100.5 and 101.0 are positions 0, 1
# and 2; 'avg_over_sent(latency, 0, 2) > 0.15' is unknown until both of the first two finish (in either order),
# averages 0.1 and 0.3 to 0.2, and ignores the third request's 9.0.
def test_dispatch_order_places_results() -> None:
    monitor = _monitor("(t >= 300) | (avg_over_sent(latency, 0, 2) > 0.15)")
    for scheduled in (100.0, 100.5, 101.0):
        monitor.dispatched(scheduled)
    monitor.observe(_result(9.0, start=101.0), now=110.0)
    monitor.observe(_result(0.3, start=100.5), now=110.5)
    assert monitor.met_at is None
    monitor.observe(_result(0.1, start=100.0), now=111.0)
    assert monitor.met_at == 11.0
    assert monitor.values_when_met == {"avg_over_sent(latency, 0, 2)": pytest.approx(0.2)}


# Two requests scheduled at the same instant get both positions, first come first served: a range of (0, 2) over
# two results scheduled at 100.0 completes.
def test_same_scheduled_time_gets_both_positions() -> None:
    monitor = _monitor("(t >= 300) | (count_over_sent(latency, 0, 2) >= 2)")
    monitor.dispatched(100.0)
    monitor.dispatched(100.0)
    monitor.observe(_result(0.1, start=100.0), now=101.0)
    monitor.observe(_result(0.1, start=100.0), now=102.0)
    assert monitor.met_at == 2.0


# Only the positions a term reads are remembered: with 'avg_over_sent(latency, 0, 2)' and 1000 dispatches, two
# scheduled times are kept; with no _over_sent term, none.
def test_only_needed_positions_are_kept() -> None:
    monitor = _monitor("(t >= 300) | (avg_over_sent(latency, 0, 2) > 1)")
    plain = _monitor()
    for i in range(1000):
        monitor.dispatched(100.0 + i)
        plain.dispatched(100.0 + i)
    assert len(monitor._sent_positions) == 2
    assert len(plain._sent_positions) == 0


# per_result_values matches the report for a streamed success of 5 tokens, 0.3s long, first token at 0.05s,
# scheduled 0.02s before it started: latency 0.3, ttft 0.05, tpot (0.3 - 0.05) / 4, ntpot 0.3 / 5 and
# schedule_delay 0.02, each equal to the report's field.
def test_per_result_values_match_the_report() -> None:
    streamed = _result(0.3, output_tokens=5, ttft=0.05, scheduled=99.98)
    report = compute_request_latency_metrics(streamed)
    load_summary = summarize_requests([streamed], [50.0]).load_summary
    values = per_result_values(streamed)
    assert values == pytest.approx(
        {
            "latency": report["request_latency"],
            "ttft": report["time_to_first_token"],
            "tpot": report["time_per_output_token"],
            "ntpot": report["normalized_time_per_output_token"],
            "schedule_delay": load_summary["schedule_delay"]["mean"],
        }
    )
    assert values["tpot"] == pytest.approx((0.3 - 0.05) / 4)
    assert values["schedule_delay"] == pytest.approx(0.02)


# A streamed response needs two chunk timestamps for ttft and two output tokens for tpot: one timestamp gives
# neither, and two timestamps with one output token give ttft but no tpot.
def test_per_result_values_streamed_edges() -> None:
    one_chunk = _result(0.3, output_tokens=1, ttft=0.05)
    assert "ttft" not in per_result_values(one_chunk)
    two_chunks_one_token = _result(0.3, output_tokens=2, ttft=0.05)
    assert isinstance(two_chunks_one_token.info.response_metrics, StreamedResponseMetrics)
    two_chunks_one_token.info.response_metrics.output_tokens = 1
    values = per_result_values(two_chunks_one_token)
    assert "ttft" in values and "tpot" not in values


# A unary success has latency and schedule_delay, no output tokens and so no ntpot; a failure has schedule_delay alone.
def test_per_result_values_unary_and_failed() -> None:
    assert set(per_result_values(_result(0.2))) == {"latency", "schedule_delay"}
    assert set(per_result_values(_result(0.2, error=True))) == {"schedule_delay"}


# With use_server_output_tokens, tpot and ntpot divide by the server's count as the report does: 5 streamed
# tokens, server says 11, so tpot is (0.3 - 0.05) / 10 and ntpot 0.3 / 11.
def test_per_result_values_follow_server_output_tokens() -> None:
    streamed = _result(0.3, output_tokens=5, ttft=0.05, server_output_tokens=11)
    report = compute_request_latency_metrics(streamed, use_server_output_tokens=True)
    values = per_result_values(streamed, use_server_output_tokens=True)
    assert values["tpot"] == pytest.approx(report["time_per_output_token"]) == pytest.approx(0.025)
    assert values["ntpot"] == pytest.approx(report["normalized_time_per_output_token"]) == pytest.approx(0.3 / 11)
