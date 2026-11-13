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
"""Acceptance tests for open-loop load under overload and the hard stop.

Two properties a load generator used to measure an overloaded server must have:

1. Open loop. A rate-based stage issues requests on its arrival schedule no
   matter how far completions lag. Nothing (a concurrency cap, a connection
   pool) may quietly turn it into a closed loop, where each new send waits for
   an old request to finish and the offered load drops to what the server can
   take.
2. Hard stop. When a stage's window ends the generator stops at once: nothing
   more is sent, every request still in flight is cancelled (not drained), and
   those requests are reported as cancelled rather than as successes,
   failures, or not at all.

What is faked and what is the oracle
------------------------------------
Faked: the server. ``AbsorberServer`` answers after a fixed delay (or never)
and records, on its own clock, when each request arrived and when the client
disconnected. That server-side record is the oracle: it does not depend on
anything inference-perf reports about itself. ``time.perf_counter()`` is
CLOCK_MONOTONIC on Linux and so shared with the inference-perf process, which
lets server arrivals be compared with the client's stage boundaries.

Each test runs the real CLI, with its worker processes, as a subprocess.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from inference_perf.apis.base import STAGE_TEARDOWN_CANCELLED_ERROR_TYPE
from utils.absorber import AbsorbedRequest, AbsorberServer
from utils.benchmark import BenchmarkResult, run_benchmark_minimal
from utils.load_shape import assert_arrivals_spread
from utils.net import get_free_port
from utils.testdata import extract_tarball

TEST_MODEL_TARBALL = "e2e/testdata/models/google_gemma-3-270m.tar.gz"

# How late a send may go out and still count as on schedule, at the 99th percentile.
SCHEDULE_DELAY_P99_LIMIT_S = 0.05
# How long after the end of a stage the client may still hold connections open.
DISCONNECT_LIMIT_S = 2.0
# Tolerance on "no arrival after the end of the stage": scheduling jitter of the
# last send plus the network hop to the server.
ARRIVAL_SLACK_S = 0.1


# The config for one run against `server`: random-free mock data, completion API
# with streaming, the given `load` block, and every lifecycle report written.
def _config(server: AbsorberServer, model_path: str, load: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "data": {"type": "mock"},
        "load": {"num_workers": 2, "request_retries": 0, **load},
        "api": {"type": "completion", "streaming": True},
        "server": {"type": "vllm", "model_name": server.model, "base_url": server.base_url, "ignore_eos": True},
        "tokenizer": {"pretrained_model_name_or_path": model_path},
        "report": {"request_lifecycle": {"summary": True, "per_stage": True, "per_request": True}},
    }


# An absorber that answers each request after `latency_s` (one chunk), or never
# when stall=True, and records client disconnects.
def _server(model_path: str, latency_s: float = 0.0, stall: bool = False) -> AbsorberServer:
    return AbsorberServer(
        port=get_free_port(),
        model=model_path,
        ttft_s=latency_s,
        itl_s=0.0,
        default_output_tokens=1,
        stall=stall,
        record_disconnects=True,
    )


# Runs the benchmark and returns (result, per-request entries, wall time when it exited).
async def _run(config: Dict[str, Any], timeout_sec: int = 240) -> Tuple[BenchmarkResult, List[Dict[str, Any]], float]:
    result = await run_benchmark_minimal(config, timeout_sec=timeout_sec)
    exited_at = time.perf_counter()
    assert result.success, f"inference-perf failed (rc={result.return_code}):\n{result.stdout[-4000:]}"
    assert result.reports is not None
    entries = result.reports["per_request_lifecycle_metrics.json"]
    return result, entries, exited_at


# Server-side end of a request: when its reply finished or the client left.
def _server_end(request: AbsorbedRequest) -> float:
    ends = [t for t in (request.done_s, request.disconnect_s) if t is not None]
    return min(ends) if ends else float("inf")


# Most requests the server held at once, from its own arrival/end record.
def _server_max_inflight(requests: List[AbsorbedRequest]) -> int:
    events = sorted([(r.arrival_s, 1) for r in requests] + [(_server_end(r), -1) for r in requests])
    level = peak = 0
    for _, delta in events:
        level += delta
        peak = max(peak, level)
    return peak


# The end of constant-rate stage `stage_id` on the shared clock. ConstantLoadTimer
# schedules a stage's last request exactly at its end, so this is the latest
# scheduled_time, valid whenever every scheduled request was recorded (sent).
def _stage_end(entries: List[Dict[str, Any]], stage_id: int) -> float:
    return max(e["scheduled_time"] for e in entries if e.get("stage_id") == stage_id)


# Everything wrong with a run that was supposed to be open-loop at `rate` for a
# window ending at `end`, as human-readable strings (empty means open-loop):
# server-side arrivals inside the window must be rate * duration within 1%, and
# client-side sends must leave on schedule (start - scheduled p99 under 50 ms).
def _open_loop_violations(
    requests: List[AbsorbedRequest], entries: List[Dict[str, Any]], rate: float, duration: float, end: float
) -> List[str]:
    problems = []
    arrivals = sum(1 for r in requests if r.arrival_s <= end + ARRIVAL_SLACK_S)
    expected = rate * duration
    if abs(arrivals - expected) > 0.01 * expected:
        problems.append(f"server saw {arrivals} arrivals in the window, expected {expected:.0f} +/- 1%")
    delays = [e["start_time"] - e["scheduled_time"] for e in entries]
    p99 = float(np.percentile(delays, 99)) if delays else float("inf")
    if p99 >= SCHEDULE_DELAY_P99_LIMIT_S:
        problems.append(f"schedule delay p99 {p99 * 1000:.0f} ms >= {SCHEDULE_DELAY_P99_LIMIT_S * 1000:.0f} ms")
    return problems


# Most server arrivals inside any window of `width` seconds. Under a truly open
# loop with a fixed server latency, everything that arrived in the last `latency`
# seconds is still in flight, so this is the expected peak in-flight count.
def _max_arrivals_in_window(requests: List[AbsorbedRequest], width: float) -> int:
    arrivals = sorted(r.arrival_s for r in requests)
    lo = peak = 0
    for hi, t in enumerate(arrivals):
        while arrivals[lo] < t - width:
            lo += 1
        peak = max(peak, hi - lo + 1)
    return peak


# Requests the server held open at time `t`, counting the last send's arrival
# if it lands within ARRIVAL_SLACK_S after `t`.
def _server_inflight_at(requests: List[AbsorbedRequest], t: float) -> int:
    return sum(1 for r in requests if r.arrival_s <= t + ARRIVAL_SLACK_S and _server_end(r) > t)


# How long a response the server has finished may sit unread in the client
# before the client processes it. A request cancelled in that window is
# cancelled for the client but finished for the server (no disconnect).
READ_LAG_S = 0.25


# Whether the client's cancelled requests agree with what the server saw
# abandoned, as None or the reason they do not. Every server-side disconnect is
# a cancelled request. The client may count more, but only:
# - requests whose reply the server finished within READ_LAG_S before the
#   client cancelled them (finished, but not yet read: no disconnect), and
# - one request scheduled exactly at the end `end`, stamped as sent and then
#   cancelled before its connection reaches the server.
# (The server's in-flight count at the nominal end can differ from both by
# requests that finish in the few ms the cancellation takes to land, which is
# why disconnects are the oracle.)
def _cancelled_mismatch(cancelled: List[Dict[str, Any]], requests: List[AbsorbedRequest], end: float) -> Optional[str]:
    disconnects = sum(1 for r in requests if r.disconnect_s is not None)
    cancelled_by = max((e["end_time"] for e in cancelled), default=end)
    unread = sum(
        1 for r in requests if r.disconnect_s is None and r.done_s is not None and end - READ_LAG_S <= r.done_s <= cancelled_by
    )
    last_send = 1 if cancelled and max(e["start_time"] for e in cancelled) >= end - ARRIVAL_SLACK_S else 0
    if disconnects <= len(cancelled) <= disconnects + unread + last_send:
        return None
    return (
        f"{len(cancelled)} cancelled on the client, {disconnects} disconnects on the server "
        f"(+ up to {unread} finished-but-unread and {last_send} unsent)"
    )


# Entries the client recorded as cancelled, and those it recorded as successes.
def _split(entries: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    cancelled = [e for e in entries if (e.get("error") or {}).get("error_type") == STAGE_TEARDOWN_CANCELLED_ERROR_TYPE]
    successes = [e for e in entries if e.get("error") is None]
    return cancelled, successes


# Test 1, open loop: 10 s server latency, 50 qps for 30 s. The server sees 1500
# +/- 1% arrivals inside the window, sends leave within 50 ms of schedule at
# p99, and about 500 requests (rate x latency) are in flight at the peak: as many
# as arrived in the busiest 10 s, so no send ever waited for a completion. With
# the hard stop, exactly the requests the server held at the end are cancelled
# and the rest succeed.
@pytest.mark.asyncio
async def test_open_loop_keeps_schedule_far_beyond_server_capacity() -> None:
    rate, duration, latency = 50, 30, 10.0
    model_path = str(extract_tarball(TEST_MODEL_TARBALL))
    async with _server(model_path, latency_s=latency) as server:
        load = {
            "type": "constant",
            "stages": [{"rate": rate, "duration": duration}],
            "open_loop": True,
            "hard_stop": True,
            "worker_max_tcp_connections": 0,
        }
        _, entries, _ = await _run(_config(server, model_path, load))
        requests = list(server.requests)

    end = _stage_end(entries, 0)
    assert _open_loop_violations(requests, entries, rate, duration, end) == []
    peak = _server_max_inflight(requests)
    assert abs(peak - _max_arrivals_in_window(requests, latency)) <= 2
    assert 0.85 * rate * latency <= peak <= 1.15 * rate * latency, f"peak in flight {peak}, expected ~{rate * latency:.0f}"
    cancelled, successes = _split(entries)
    assert len(cancelled) + len(successes) == len(entries)
    assert (mismatch := _cancelled_mismatch(cancelled, requests, end)) is None, mismatch


# Test 2, stalled server: the server never answers. 50 qps for 10 s still puts
# 500 +/- 1% arrivals on the server, spread over the window like the timer's own
# schedule (each 2 s slice within 4.5 sigma of 100), and with the hard stop all
# 500 are reported cancelled.
@pytest.mark.asyncio
async def test_arrivals_follow_schedule_when_server_never_answers() -> None:
    rate, duration = 50, 10
    model_path = str(extract_tarball(TEST_MODEL_TARBALL))
    async with _server(model_path, stall=True) as server:
        load = {
            "type": "constant",
            "stages": [{"rate": rate, "duration": duration}],
            "open_loop": True,
            "hard_stop": True,
            "worker_max_tcp_connections": 0,
        }
        _, entries, _ = await _run(_config(server, model_path, load))
        requests = list(server.requests)

    assert _open_loop_violations(requests, entries, rate, duration, _stage_end(entries, 0)) == []
    assert_arrivals_spread([{"start_time": r.arrival_s} for r in requests])
    cancelled, successes = _split(entries)
    assert successes == [] and len(cancelled) == len(entries)
    assert (mismatch := _cancelled_mismatch(cancelled, requests, _stage_end(entries, 0))) is None, mismatch


# Test 3, cap detector: the default bounded dispatch with 2 workers x 50 permits
# caps in-flight requests at 100. Against the 10 s server at 50 qps, sends then
# go out late and far fewer than 750 arrive in 15 s, and the open-loop check of
# test 1 flags it: this proves the check detects a closed loop.
@pytest.mark.asyncio
async def test_open_loop_check_detects_a_concurrency_cap() -> None:
    rate, duration, latency = 50, 15, 10.0
    model_path = str(extract_tarball(TEST_MODEL_TARBALL))
    async with _server(model_path, latency_s=latency) as server:
        load = {
            "type": "constant",
            "stages": [{"rate": rate, "duration": duration}],
            "worker_max_concurrency": 50,
            "hard_stop": True,
        }
        _, entries, _ = await _run(_config(server, model_path, load))
        requests = list(server.requests)

    # Not every scheduled request was sent, so the end is taken from the first
    # send rather than the last; the arrival shortfall dwarfs that approximation.
    end = min(e["scheduled_time"] for e in entries) + duration
    problems = _open_loop_violations(requests, entries, rate, duration, end)
    assert any("arrivals" in p for p in problems), problems
    assert _server_max_inflight(requests) <= 100


# Test 4, hard stop: 120 s server latency, 10 qps for 30 s. No request reaches
# the server after the stage ends at T, every client connection is closed by
# T + 2 s, the process exits on its own, and the cancelled count equals the
# requests in flight at T (all of them), none of which enter latency stats.
@pytest.mark.asyncio
async def test_hard_stop_cancels_everything_in_flight_at_stage_end() -> None:
    rate, duration, latency = 10, 30, 120.0
    model_path = str(extract_tarball(TEST_MODEL_TARBALL))
    async with _server(model_path, latency_s=latency) as server:
        load = {
            "type": "constant",
            "stages": [{"rate": rate, "duration": duration}],
            "open_loop": True,
            "hard_stop": True,
            "worker_max_tcp_connections": 0,
        }
        result, entries, exited_at = await _run(_config(server, model_path, load))
        requests = list(server.requests)

    end = _stage_end(entries, 0)
    late = [r.arrival_s - end for r in requests if r.arrival_s > end + ARRIVAL_SLACK_S]
    assert late == [], f"{len(late)} arrivals after the stage ended"
    lingering = [_server_end(r) - end for r in requests if _server_end(r) > end + DISCONNECT_LIMIT_S]
    assert lingering == [], f"{len(lingering)} connections still open {DISCONNECT_LIMIT_S} s after the end"
    assert exited_at - end < 60, f"process exited {exited_at - end:.1f} s after the stage ended"
    cancelled, successes = _split(entries)
    assert _server_inflight_at(requests, end) == len(requests)
    assert len(cancelled) == len(entries)
    assert (mismatch := _cancelled_mismatch(cancelled, requests, end)) is None, mismatch
    summary = result.reports["summary_lifecycle_metrics.json"] if result.reports else {}
    assert summary["cancelled"]["count"] == len(cancelled)
    assert summary["successes"]["count"] == 0
    assert summary["successes"]["latency"]["request_latency"] is None


# Test 5, multi-stage: 20 qps then 40 qps, 5 s each with a 2 s interval, 3 s
# server latency. No stage-k request is sent or reaches the server after stage
# k ends, each stage sends rate x duration +/- 1%, and stage 0's leftovers are
# cancelled before stage 1 starts.
@pytest.mark.asyncio
async def test_no_request_crosses_a_stage_boundary() -> None:
    rates, duration, latency = [20, 40], 5, 3.0
    model_path = str(extract_tarball(TEST_MODEL_TARBALL))
    async with _server(model_path, latency_s=latency) as server:
        load = {
            "type": "constant",
            "stages": [{"rate": r, "duration": duration} for r in rates],
            "interval": 2,
            "open_loop": True,
            "hard_stop": True,
            "worker_max_tcp_connections": 0,
        }
        _, entries, _ = await _run(_config(server, model_path, load))
        requests = list(server.requests)

    end0 = _stage_end(entries, 0)
    start1 = min(e["scheduled_time"] for e in entries if e["stage_id"] == 1)
    for stage_id, rate in enumerate(rates):
        stage_entries = [e for e in entries if e["stage_id"] == stage_id]
        end = _stage_end(entries, stage_id)
        assert abs(len(stage_entries) - rate * duration) <= 0.01 * rate * duration
        assert max(e["start_time"] for e in stage_entries) <= end + ARRIVAL_SLACK_S
    assert not [r for r in requests if end0 + ARRIVAL_SLACK_S < r.arrival_s < start1 - ARRIVAL_SLACK_S]
    stage0_requests = [r for r in requests if r.arrival_s <= end0 + ARRIVAL_SLACK_S]
    assert all(_server_end(r) <= end0 + DISCONNECT_LIMIT_S for r in stage0_requests)
    assert all(_server_end(r) < start1 for r in stage0_requests)


# Test 6, fixed concurrency C=8 with a 5.25 s time bound, 0.5 s server latency:
# the server never holds more than 8 (and does hold 8), throughput is about
# C / latency = 16 req/s, nothing arrives after the bound, the requests in
# flight at the bound (at most 8; a slot can be between requests at that
# instant) are cancelled and their connections close within 2 s. The 8 slots
# move in lockstep, so a bound that is a multiple of the latency would land on
# a completion wave: responses the server has finished but the client has not
# yet read are cancelled, and count as cancelled with no server disconnect.
# The extra 0.25 s puts the bound mid-wave.
@pytest.mark.asyncio
async def test_fixed_concurrency_holds_exactly_c_and_stops_at_its_bound() -> None:
    concurrency, bound, latency = 8, 5.25, 0.5
    model_path = str(extract_tarball(TEST_MODEL_TARBALL))
    async with _server(model_path, latency_s=latency) as server:
        load = {
            "type": "concurrent",
            "stages": [{"num_requests": 10000, "concurrency_level": concurrency, "max_duration": bound}],
            "hard_stop": True,
        }
        _, entries, _ = await _run(_config(server, model_path, load))
        requests = list(server.requests)

    assert _server_max_inflight(requests) == concurrency
    start = min(r.arrival_s for r in requests)
    # The bound runs from the stage's schedule start, which is the first send.
    end = min(e["scheduled_time"] for e in entries) + bound
    completed_in_window = sum(1 for r in requests if r.done_s is not None and r.disconnect_s is None)
    throughput = completed_in_window / (end - start)
    assert abs(throughput - concurrency / latency) <= 0.15 * concurrency / latency, f"{throughput:.1f} req/s"
    assert not [r for r in requests if r.arrival_s > end + ARRIVAL_SLACK_S]
    assert all(_server_end(r) <= end + DISCONNECT_LIMIT_S for r in requests)
    cancelled, _ = _split(entries)
    assert 0 < len(cancelled) <= concurrency
    # Cancelled at the bound, not a permit-poll interval later.
    assert all(e["end_time"] <= end + ARRIVAL_SLACK_S for e in cancelled)
    assert (mismatch := _cancelled_mismatch(cancelled, requests, end)) is None, mismatch


# Pins the helper the tests above rely on: two requests overlapping in [1, 2)
# plus one alone in [3, 4) give a peak of 2.
def test_server_max_inflight_counts_overlap() -> None:
    def req(arrival: float, done: Optional[float]) -> AbsorbedRequest:
        return AbsorbedRequest(route="/v1/completions", arrival_s=arrival, body={}, done_s=done)

    assert _server_max_inflight([req(0.0, 2.0), req(1.0, 2.5), req(3.0, 4.0)]) == 2
