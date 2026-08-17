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
"""Pins the reported load-side values to hand-computable known inputs.

`test_lifecycle_report_shape.py` already reaches `load_summary`, but it only asserts
shape: that `schedule_delay` carries mean/min/median/max and that `achieved_rate > 0`.
Both are exactly computable from the fixture it builds. This module supplies request
lifecycle metrics with explicitly chosen scheduled and actual send times and asserts the
reported numbers equal the values those times imply.

Every expected value below is written as a literal or as a one-step arithmetic
expression over the injected times. Nothing here recomputes an expectation by calling
the code under test, so a change in `summarize_requests` cannot move the oracle with it.

Oracle: the injected `scheduled_time` and `start_time` values in each test.
Faked: only the lifecycle metric objects. `summarize_requests` itself is the real one.
"""

import typing
from unittest.mock import Mock

import pytest

from inference_perf.apis import ErrorResponseInfo, InferenceInfo
from inference_perf.payloads import RequestMetrics, Text
from inference_perf.reportgen.base import summarize_requests
from test_lifecycle_report_shape import _mock_metric

# Percentile list wide enough to cover both key-naming branches in `summarize`
# (p == 50 becomes "median", everything else becomes "p<n>") and to exercise
# numpy's linear interpolation between samples.
PERCENTILES: typing.List[float] = [50, 90, 99]


def _load_metric(*, scheduled_time: float, start_time: float, end_time: float) -> Mock:
    """A lifecycle metric whose load-side timing is the only interesting thing about it.

    Delegates to the shape test's builder so both modules describe a request the same
    way. Token and modality fields are filled with the smallest values that keep the
    success path of `summarize_requests` well-defined; no assertion here reads them.
    """
    return _mock_metric(
        start_time=start_time,
        end_time=end_time,
        scheduled_time=scheduled_time,
        input_tokens=10,
        output_tokens=2,
        request_data="x",
        images=[],
        videos=[],
        audios=[],
        output_token_times=[start_time + 0.01, end_time],
    )


def _delayed(*, start_times: typing.Sequence[float], delays: typing.Sequence[float]) -> typing.List[Mock]:
    """Requests that were sent at `start_times` after waiting `delays` in the client queue.

    `scheduled_time` is derived as `start_time - delay`, which is the inverse of the
    `start_time - scheduled_time` that `summarize_requests` computes, so the expected
    schedule delay of request i is exactly `delays[i]`.
    """
    assert len(start_times) == len(delays)
    return [
        _load_metric(scheduled_time=start - delay, start_time=start, end_time=start + 0.5)
        for start, delay in zip(start_times, delays, strict=True)
    ]


def _load_summary(metrics: typing.List[Mock], **kwargs: typing.Any) -> typing.Dict[str, typing.Any]:
    summary = summarize_requests(typing.cast(typing.Any, metrics), percentiles=PERCENTILES, **kwargs)
    return typing.cast(typing.Dict[str, typing.Any], summary.model_dump()["load_summary"])


def test_schedule_delay_is_zero_when_every_request_left_on_time() -> None:
    """A load generator that never falls behind must report a flat zero delay.

    This is the reading that says a latency number can be trusted, so it has to be
    exactly 0.0 rather than merely small.
    """
    load = _load_summary(_delayed(start_times=[0.0, 1.0, 2.0], delays=[0.0, 0.0, 0.0]), stage_rate=1.0)

    assert load["count"] == 3
    assert load["schedule_delay"] == {"mean": 0.0, "min": 0.0, "max": 0.0, "median": 0.0, "p90": 0.0, "p99": 0.0}


def test_schedule_delay_matches_a_uniform_injected_delay() -> None:
    """Every request held 0.25s: every reported statistic must be 0.25s."""
    load = _load_summary(
        _delayed(start_times=[0.25, 1.25, 2.25, 3.25], delays=[0.25, 0.25, 0.25, 0.25]),
        stage_rate=1.0,
    )

    assert load["count"] == 4
    for key in ("mean", "min", "max", "median", "p90", "p99"):
        assert load["schedule_delay"][key] == pytest.approx(0.25), key


def test_schedule_delay_percentiles_match_mixed_injected_delays() -> None:
    """Five known delays, supplied out of order, checked against hand arithmetic.

    Injected delays, as a set: 0.0, 0.1, 0.2, 0.3, 0.4. Sorted, that is
    [0.0, 0.1, 0.2, 0.3, 0.4], so with numpy's default linear interpolation over
    index positions 0..4:
      mean   = (0.0 + 0.1 + 0.2 + 0.3 + 0.4) / 5 = 0.2
      min    = 0.0
      max    = 0.4
      median = value at index 0.50 * 4 = 2.0             -> 0.2
      p90    = value at index 0.90 * 4 = 3.6             -> 0.3 + 0.6 * 0.1 = 0.36
      p99    = value at index 0.99 * 4 = 3.96            -> 0.3 + 0.96 * 0.1 = 0.396

    The delays are attached to send times in a deliberately non-monotone order to show
    the reported distribution does not depend on the order requests were observed in.
    """
    load = _load_summary(
        _delayed(start_times=[0.0, 1.0, 2.0, 3.0, 4.0], delays=[0.2, 0.0, 0.4, 0.1, 0.3]),
        stage_rate=1.0,
    )

    assert load["count"] == 5
    assert load["schedule_delay"]["mean"] == pytest.approx(0.2)
    assert load["schedule_delay"]["min"] == pytest.approx(0.0)
    assert load["schedule_delay"]["max"] == pytest.approx(0.4)
    assert load["schedule_delay"]["median"] == pytest.approx(0.2)
    assert load["schedule_delay"]["p90"] == pytest.approx(0.36)
    assert load["schedule_delay"]["p99"] == pytest.approx(0.396)


def test_schedule_delay_includes_failed_requests() -> None:
    """The load population is every dispatched request, not just the successful ones.

    A request that was held in the client queue and then failed is still evidence the
    generator fell behind, so dropping it would flatter the delay distribution exactly
    when the run is least trustworthy. Two successes delayed 0.0s and 0.2s plus one
    failure delayed 1.0s give mean = 1.2 / 3 = 0.4 and max = 1.0; if failures were
    excluded the mean would be 0.1 and the max 0.2.
    """
    success_a, success_b = _delayed(start_times=[0.0, 1.0], delays=[0.0, 0.2])

    failure = Mock()
    failure.scheduled_time = 1.0
    failure.start_time = 2.0
    failure.end_time = 2.5
    failure.error = ErrorResponseInfo(error_type="HTTP Error 500", error_msg="Internal Server Error")
    failure.session_id = None
    failure.ttft_slo_sec = None
    failure.tpot_slo_sec = None
    failure.request_data = "bad"
    failure.info = Mock(spec=InferenceInfo)
    failure.info.request_metrics = RequestMetrics(text=Text(input_tokens=10))
    failure.info.response_metrics = None
    failure.info.extra_info = {}

    load = _load_summary([success_a, success_b, failure], stage_rate=1.0)

    assert load["count"] == 3
    assert load["schedule_delay"]["mean"] == pytest.approx(0.4)
    assert load["schedule_delay"]["min"] == pytest.approx(0.0)
    assert load["schedule_delay"]["max"] == pytest.approx(1.0)


def test_achieved_rate_equals_count_over_send_duration() -> None:
    """Five requests sent over a 4.0s window report 5 / 4.0 = 1.25 per second.

    Note what the reported number is not: those five requests were dispatched at exactly
    1.0s intervals, so the rate the generator actually sustained is 1.0/s. The report
    divides the request count, not the number of gaps between requests, by the span from
    the first send to the last, which overstates the sustained rate by a factor of
    n / (n - 1). This asserts the shipped definition rather than the intended one; see
    the PR description.
    """
    load = _load_summary(
        _delayed(start_times=[0.0, 1.0, 2.0, 3.0, 4.0], delays=[0.0, 0.0, 0.0, 0.0, 0.0]),
        stage_rate=1.0,
    )

    assert load["count"] == 5
    assert load["requested_rate"] == 1.0
    assert load["send_duration"] == pytest.approx(4.0)
    assert load["achieved_rate"] == pytest.approx(1.25)


def test_achieved_rate_tracks_send_duration_not_completion_times() -> None:
    """Slow responses must not drag the achieved rate down.

    Sends are 0.5s apart, so the four requests span 1.5s and the reported rate is
    4 / 1.5. Each request then takes 10s to come back, which moves every end_time but
    must leave the load-side rate untouched: `achieved_rate` describes dispatch, and
    conflating it with completion would hide a generator that is keeping up against a
    server that is not.
    """
    metrics = [_load_metric(scheduled_time=start, start_time=start, end_time=start + 10.0) for start in (0.0, 0.5, 1.0, 1.5)]

    load = _load_summary(metrics, stage_rate=8.0)

    assert load["send_duration"] == pytest.approx(1.5)
    assert load["achieved_rate"] == pytest.approx(4.0 / 1.5)
    assert load["requested_rate"] == 8.0


def test_achieved_rate_is_zero_for_a_single_request() -> None:
    """One request gives a zero-width send window, and the shipped guard reports 0.0.

    `summarize_requests` divides by `send_duration` only when it is positive and
    substitutes 0.0 otherwise. That sentinel is indistinguishable from a stage that
    genuinely achieved no throughput, so this pins the current behavior rather than
    endorsing it; see the PR description.
    """
    load = _load_summary(_delayed(start_times=[7.0], delays=[0.3]), stage_rate=2.0)

    assert load["count"] == 1
    assert load["send_duration"] == 0.0
    assert load["achieved_rate"] == 0.0
    # The delay is still exact with a single sample.
    assert load["schedule_delay"] == {
        "mean": pytest.approx(0.3),
        "min": pytest.approx(0.3),
        "max": pytest.approx(0.3),
        "median": pytest.approx(0.3),
        "p90": pytest.approx(0.3),
        "p99": pytest.approx(0.3),
    }


def test_achieved_rate_is_zero_when_all_requests_share_a_send_time() -> None:
    """The other zero-width window: a burst dispatched at one instant.

    Three requests sent at the same timestamp is the closest thing to infinite achieved
    rate, and it reports 0.0. Same guard, same caveat as the single-request case.
    """
    load = _load_summary(
        _delayed(start_times=[5.0, 5.0, 5.0], delays=[0.1, 0.1, 0.1]),
        stage_rate=100.0,
    )

    assert load["count"] == 3
    assert load["send_duration"] == 0.0
    assert load["achieved_rate"] == 0.0


def test_load_summary_omits_rate_fields_without_a_stage_rate() -> None:
    """No requested rate means no rate block at all, not a rate reported as zero.

    A stage driven by concurrency rather than by a rate has no `achieved_rate` key, so a
    consumer reading it unconditionally gets a KeyError rather than a wrong number.
    `schedule_delay` is still reported and still exact.
    """
    load = _load_summary(_delayed(start_times=[0.0, 1.0, 2.0], delays=[0.5, 0.5, 0.5]))

    assert load["count"] == 3
    assert load["schedule_delay"]["mean"] == pytest.approx(0.5)
    assert "achieved_rate" not in load
    assert "send_duration" not in load
    assert "requested_rate" not in load


def test_schedule_delay_is_signed_when_a_request_leaves_early() -> None:
    """A send that beats its schedule reports a negative delay, and it is not clamped.

    The shape fixture in `test_lifecycle_report_shape.py` already contains one of these
    (`scheduled_time=-0.001` against `start_time=0.0`), so the sign convention is
    reachable in practice and worth pinning: three requests early by 0.01s, on time, and
    late by 0.04s give mean = (-0.01 + 0.0 + 0.04) / 3 = 0.01 and min = -0.01. A clamp
    at zero would report mean = 0.04 / 3 and hide the early sends entirely.
    """
    load = _load_summary(
        _delayed(start_times=[0.0, 1.0, 2.0], delays=[-0.01, 0.0, 0.04]),
        stage_rate=1.0,
    )

    assert load["schedule_delay"]["min"] == pytest.approx(-0.01)
    assert load["schedule_delay"]["max"] == pytest.approx(0.04)
    assert load["schedule_delay"]["mean"] == pytest.approx(0.01)
