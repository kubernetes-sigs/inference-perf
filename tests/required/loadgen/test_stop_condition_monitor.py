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
condition on several consecutive evaluations. The values it feeds the terms
are the report's, and the levels come from the load generator's phases.
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
from inference_perf.apis.base import stage_teardown_cancelled_error
from inference_perf.loadgen.stop_condition import StopConditionMonitor
from inference_perf.payloads import RequestMetrics, Text
from inference_perf.reportgen.base import series_values
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


# The load generator's requests_by_phase() with the given counts, 0 for a phase left out.
def _phases(queued: int = 0, waiting: int = 0, streaming: int = 0) -> dict[str, int]:
    return {"queued": queued, "waiting": waiting, "streaming": streaming}


# A result the load generator cancelled at stage teardown, ended 0.1s after it started at clock 100.
def _cancelled() -> RequestLifecycleMetric:
    metric = _result()
    metric.error = stage_teardown_cancelled_error()
    return metric


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
    monitor.tick(105.0, phases=_phases(waiting=2))
    assert monitor.met_at is None
    monitor.tick(106.0, phases=_phases(waiting=2))
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


# A term reads the report's number. A streamed success of 5 tokens, 0.3s long, first token at 0.05s: the value
# 'avg_over_last(ttft, 1)' ends the stage on is series_values' ttft, 0.05.
def test_term_reads_the_report_value() -> None:
    streamed = _result(0.3, output_tokens=5, ttft=0.05)
    monitor = _monitor("(t >= 300) | (avg_over_last(ttft, 1) > 0.01)")
    monitor.observe(streamed, now=101.0)
    assert monitor.met_at == 1.0
    assert monitor.values_when_met == {"avg_over_last(ttft, 1)": pytest.approx(series_values(streamed)["ttft"])}
    assert series_values(streamed)["ttft"] == pytest.approx(0.05)


# A response that arrived in one chunk has a ttft in the report, so it has one here: 'avg_over_last(ttft, 1) > 0.01'
# ends the stage on a one-token stream whose only chunk landed 0.05s in.
def test_one_chunk_stream_has_a_ttft() -> None:
    monitor = _monitor("(t >= 300) | (avg_over_last(ttft, 1) > 0.01)")
    monitor.observe(_result(0.3, output_tokens=1, ttft=0.05), now=101.0)
    assert monitor.met_at == 1.0


# With use_server_output_tokens, ntpot divides by the server's count as the report does: 5 streamed tokens,
# server says 11, 0.3s long, so 'avg_over_last(ntpot, 1)' reads 0.3 / 11.
def test_term_follows_server_output_tokens() -> None:
    monitor = StopConditionMonitor(
        0, Predicate("(t >= 300) | (avg_over_last(ntpot, 1) > 0)"), hold=1, use_server_output_tokens=True
    )
    monitor.start(100.0)
    monitor.observe(_result(0.3, output_tokens=5, ttft=0.05, server_output_tokens=11), now=101.0)
    assert monitor.values_when_met == {"avg_over_last(ntpot, 1)": pytest.approx(0.3 / 11)}


# A request cancelled at teardown is not a failure: three of them leave 'errors >= 3' false, and
# 'completed{status="cancelled"} >= 3' holds on the third.
def test_cancelled_result_is_not_an_error() -> None:
    errors = _monitor("(t >= 300) | (errors >= 3)")
    cancelled = _monitor('(t >= 300) | (completed{status="cancelled"} >= 3)')
    for second in (101.0, 102.0, 103.0):
        errors.observe(_cancelled(), now=second)
        cancelled.observe(_cancelled(), now=second)
    assert errors.met_at is None
    assert cancelled.met_at == 3.0


# 'completed offset 5s >= 2' reads the count as it was 5s ago. Two results finish at stage second 0.1, observed at
# clock 101 and 102: no value yet. The tick at clock 106 (t=6, so t-5=1 is past both) ends the stage.
def test_offset_reads_the_count_as_it_was() -> None:
    monitor = _monitor("(t >= 300) | (completed offset 5s >= 2)")
    monitor.observe(_result(), now=101.0)
    monitor.observe(_result(), now=102.0)
    assert monitor.met_at is None
    monitor.tick(106.0)
    assert monitor.met_at == 6.0


# The levels come from the phases passed in: with 4 queued, 1 waiting and 2 streaming, 'queued >= 4' and
# 'in_flight{phase="streaming"} >= 2' each hold on the first tick, and bare in_flight reads 3, waiting plus streaming.
@pytest.mark.parametrize(
    "condition, reading",
    [
        ("(t >= 300) | (queued >= 4)", {"queued": 4.0}),
        ('(t >= 300) | (in_flight{phase="streaming"} >= 2)', {'in_flight{phase="streaming"}': 2.0}),
        ("(t >= 300) | (in_flight >= 3)", {"in_flight": 3.0}),
    ],
)
def test_levels_read_the_phases(condition: str, reading: dict[str, float]) -> None:
    monitor = _monitor(condition)
    monitor.tick(101.0, phases=_phases(queued=4, waiting=1, streaming=2))
    assert monitor.met_at == 1.0
    assert monitor.values_when_met == reading


# With no phases passed, a level has no value and the condition does not hold on it.
def test_level_without_phases_has_no_value() -> None:
    monitor = _monitor("(t >= 300) | (queued >= 0)")
    monitor.tick(101.0)
    assert monitor.met_at is None
