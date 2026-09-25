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
"""Sim-backed load-shape accuracy e2e (#633).

Everything else in the e2e tier checks what inference-perf *reports* about
the responses it got. This checks the stimulus: that the load actually
offered matches the load that was configured. For a load generator that is
the more fundamental correctness claim, and `docs/loadgen.md` makes it
explicitly ("workers are dynamically allocated to achieve the exact
concurrency specified") without anything verifying it.

What is faked and what is the oracle
------------------------------------
Faked: the server. `llm-d-inference-sim` stands in for vLLM, with pinned
latencies so request duration is known and the run is short.

Oracle: the configured load, which is a known-good input, plus a
reconstruction of delivered rate, arrival spread and in-flight concurrency
computed in `utils.load_shape` from the raw per-request
`start_time`/`end_time` pairs. The reconstruction does not read reportgen's
summary numbers, so the reported `achieved_rate` can be checked against it
rather than merely restated.

This stays on the sim deliberately. Against a real vLLM these assertions
would measure the server's capacity, not the load generator: a slow server
delays responses, and under a concurrency limit that changes the offered
load itself.

Scope, versus the neighbouring issues
-------------------------------------
Dispatch scheduling arithmetic (what times the timers emit, how a
concurrency level is split across workers) is unit-level and belongs to
#659. What only a real process with real sockets can show is whether the
generator keeps up with its own schedule and holds its semaphore under load,
so that is all this file asserts.

Known limit of the oracle
-------------------------
The timestamps come from the client, so this measures the load generator's
own view of what it offered. That is enough to catch a generator that falls
behind, a semaphore that admits the wrong number of requests, and a reportgen
that derives the rate wrongly, but not a bug in the timestamping itself. A
server that recorded arrivals independently would be strictly stronger and
this test can be retargeted at one later without changing its assertions.
"""

import pytest

from utils.accuracy import assert_successful_run
from utils.benchmark import run_benchmark_minimal
from utils.llm_d_inference_sim import LLMDInferenceSimRunner
from utils.load_shape import (
    arrival_bin_counts,
    assert_arrivals_spread,
    assert_delivered_concurrency,
    fraction_at_level,
    inflight_segments,
    max_inflight,
    mean_inflight,
    observed_send_rate,
    plateau_window,
    rate_tolerance,
    spread_tolerance,
)
from utils.net import get_free_port
from utils.testdata import extract_tarball

TEST_MODEL_NAME = "google/gemma-3-270m"
TEST_MODEL_TARBALL = "e2e/testdata/models/google_gemma-3-270m.tar.gz"

# Rate stage: 600 requests is the smallest count at which a Poisson stage can
# be held to a tolerance worth gating on (see rate_tolerance), and 40 qps is
# already exercised by the existing sim suite (test_llm_d_inference_sim runs
# 100 qps), so the client is not the bottleneck.
RATE = 40
DURATION = 15
EXPECTED_RATE_REQUESTS = RATE * DURATION

# Concurrency stages: pinned sim latencies put each request at roughly
# 150ms + 15 * 15ms = 375ms, so 12 rounds of requests is about 4.5s of load.
SIM_TTFT_MS = 150
SIM_ITL_MS = 15
OUTPUT_TOKENS = 16
ROUNDS = 12


# Sim flags every test here passes, plus any extras the caller adds.
def _sim_args(*extra: str) -> list[str]:
    return [
        # Default max-num-seqs is 5. Server-side queueing does not change what
        # the client offers, but it does stretch request duration and with it
        # the length of the run, so keep every request scheduled immediately.
        *("--max-num-seqs", "64"),
        *extra,
    ]


# The `server:` block of a config pointed at the running sim.
def _server_block(sim: LLMDInferenceSimRunner, model_name: str) -> dict:
    return {
        "type": "vllm",
        "model_name": model_name,
        "base_url": f"http://{sim.host}:{sim.port}",
        "ignore_eos": True,
    }


# Sim flags for the concurrency tests: fixed TTFT and ITL with zero jitter.
# Request duration must not vary, or the length of the plateau moves with it.
def _pinned_latency_sim_args() -> list[str]:
    return _sim_args(
        *("--time-to-first-token", str(SIM_TTFT_MS)),
        *("--inter-token-latency", str(SIM_ITL_MS)),
        *("--time-to-first-token-std-dev", "0"),
        *("--inter-token-latency-std-dev", "0"),
        *("--seed", "42"),
    )


# One concurrent stage of `num_requests` requests at `concurrency` across
# `num_workers` workers, with the given `data:` block and per-request output.
def _concurrent_stage_config(
    sim: LLMDInferenceSimRunner,
    model_name: str,
    model_path: str,
    data: dict,
    concurrency: int,
    num_workers: int,
    num_requests: int,
) -> dict:
    return {
        "data": data,
        "load": {
            "type": "concurrent",
            "stages": [{"num_requests": num_requests, "concurrency_level": concurrency}],
            "num_workers": num_workers,
        },
        "api": {"type": "completion", "streaming": True},
        "server": _server_block(sim, model_name),
        "tokenizer": {"pretrained_model_name_or_path": str(model_path)},
        "report": {
            "request_lifecycle": {
                "summary": True,
                "per_stage": True,
                "per_request": True,
            },
        },
    }


# 600 requests at a configured 40 qps for 15s, constant and poisson. Pins
# count=600 and requested_rate=40, the reported send_duration/achieved_rate
# equal to the values recomputed from per-request starts, the achieved rate
# within rate_tolerance of 40, and the arrivals spread across the window.
@pytest.mark.asyncio
@pytest.mark.skipif(not LLMDInferenceSimRunner.is_available(), reason="local environment missing llm-d-inference-sim")
@pytest.mark.parametrize("arrival", ["constant", "poisson"])
async def test_achieved_rate_matches_configured_rate(arrival: str):
    """A fixed-rate stage must deliver the rate it was configured with.

    Three separate claims, deliberately not collapsed into one:

    1. reportgen's `achieved_rate` is the count over the span of send times,
       recomputed here from the raw per-request timestamps. Checking the
       reported value against the independent recomputation is what makes
       this more than a restatement of the tool's own summary.
    2. that rate matches the configured `rate` within a tolerance derived
       from the arrival process (see `rate_tolerance`), not from a number
       chosen after watching a run.
    3. the requests were spread across the stage. Claims 1 and 2 only look
       at the first and last send, so a generator that stalled and then
       sent everything in a burst would still pass them.
    """
    model_name = TEST_MODEL_NAME
    model_path = extract_tarball(TEST_MODEL_TARBALL)

    async with LLMDInferenceSimRunner(model_name, *_sim_args(), port=get_free_port()) as sim:
        result = await run_benchmark_minimal(
            {
                "data": {"type": "mock"},
                "load": {
                    "type": arrival,
                    "stages": [{"rate": RATE, "duration": DURATION}],
                    "num_workers": 2,
                },
                "api": {"type": "completion", "streaming": True},
                "server": _server_block(sim, model_name),
                "tokenizer": {"pretrained_model_name_or_path": str(model_path)},
                "report": {
                    "request_lifecycle": {
                        "summary": True,
                        "per_stage": True,
                        "per_request": True,
                    },
                },
            },
            timeout_sec=180,
        )

    entries = assert_successful_run(result, EXPECTED_RATE_REQUESTS)

    stage = result.reports["stage_0_lifecycle_metrics.json"]["load_summary"]
    assert stage["count"] == EXPECTED_RATE_REQUESTS
    assert stage["requested_rate"] == RATE, f"stage echoed requested_rate {stage['requested_rate']}, configured {RATE}"

    # (1) reportgen's derivation, against the same quantity recomputed from
    # the raw per-request send times. Exact: it is the same arithmetic on the
    # same data, so any drift here is a reportgen bug, not timing noise.
    send_duration, recomputed_rate = observed_send_rate(entries)
    assert stage["send_duration"] == pytest.approx(send_duration, rel=1e-9), (
        f"reported send_duration {stage['send_duration']} != {send_duration} recomputed from per-request starts"
    )
    assert stage["achieved_rate"] == pytest.approx(recomputed_rate, rel=1e-9), (
        f"reported achieved_rate {stage['achieved_rate']} != {recomputed_rate} recomputed from per-request starts"
    )

    # (2) the load-shape claim itself.
    tolerance = rate_tolerance(EXPECTED_RATE_REQUESTS, arrival)
    error = abs(recomputed_rate - RATE) / RATE
    assert error <= tolerance, (
        f"{arrival} stage delivered {recomputed_rate:.3f} req/s against a configured {RATE} req/s "
        f"({error:.1%} off, tolerance {tolerance:.1%} for n={EXPECTED_RATE_REQUESTS})"
    )

    # (3) the rate held throughout, not just on average between the first
    # and last send.
    assert_arrivals_spread(entries)


# concurrency_level 8 and 5 across 2 workers, 12 rounds each. Pins the
# reported concurrency field to the configured value, in-flight never above
# it, and time-weighted in-flight over the plateau within half a slot of it.
@pytest.mark.asyncio
@pytest.mark.skipif(not LLMDInferenceSimRunner.is_available(), reason="local environment missing llm-d-inference-sim")
@pytest.mark.parametrize(
    ("concurrency", "num_workers"),
    [
        pytest.param(8, 2, id="c8_w2_divisible"),
        # concurrency_level % num_workers != 0: workers get 3 and 2. The split
        # arithmetic is unit-tested, but nothing has checked that the sum of
        # the split semaphores is what actually goes on the wire.
        pytest.param(5, 2, id="c5_w2_remainder"),
    ],
)
async def test_delivered_concurrency_matches_configured_level(concurrency: int, num_workers: int):
    """A fixed-concurrency stage must hold exactly `concurrency_level` in flight.

    `load_summary` carries a `concurrency` field, but reportgen copies it
    straight off the stage config, so asserting on it proves only that the
    value was threaded through. The delivered value has to be reconstructed
    from the per-request start and end timestamps; that reconstruction is
    the actual oracle and `assert_delivered_concurrency` explains its window.

    `achieved_rate` is not asserted for this load type. `inference_perf/main.py`
    rewrites a concurrent stage to `rate=num_requests, duration=1`, so the
    whole stage is enqueued over one second and the workers' semaphores set
    the pace. `requested_rate` on a concurrent stage is therefore a dispatch
    detail, not a load-shape claim.
    """
    model_name = TEST_MODEL_NAME
    model_path = extract_tarball(TEST_MODEL_TARBALL)
    num_requests = concurrency * ROUNDS
    data = {
        "type": "synthetic",
        "input_distribution": {"type": "fixed", "min": 32, "max": 32, "mean": 32},
        "output_distribution": {"type": "fixed", "min": OUTPUT_TOKENS, "max": OUTPUT_TOKENS, "mean": OUTPUT_TOKENS},
    }

    async with LLMDInferenceSimRunner(model_name, *_pinned_latency_sim_args(), port=get_free_port()) as sim:
        result = await run_benchmark_minimal(
            _concurrent_stage_config(sim, model_name, model_path, data, concurrency, num_workers, num_requests),
            timeout_sec=180,
        )

    entries = assert_successful_run(result, num_requests)

    # Wiring check only, called out as such: this field is an echo of config.
    stage = result.reports["stage_0_lifecycle_metrics.json"]["load_summary"]
    assert stage["concurrency"] == concurrency, f"stage echoed concurrency {stage['concurrency']}, configured {concurrency}"

    # The load-shape claim: what was actually in flight.
    assert_delivered_concurrency(entries, concurrency)


# concurrency_level 8 across 2 workers, but every request is pinned to worker
# 0 (shared_prefix multi-turn with one group). Pins the same delivered
# concurrency of 8: the split must follow the pins, not the worker count.
@pytest.mark.asyncio
@pytest.mark.skipif(not LLMDInferenceSimRunner.is_available(), reason="local environment missing llm-d-inference-sim")
async def test_delivered_concurrency_holds_under_pinned_routing():
    """Pinned routing must not strand part of `concurrency_level` on idle workers.

    Datagens that need worker affinity set `preferred_worker_id`, and the
    load generator routes each such request to `id % num_workers`. If the
    pins reach fewer workers than `num_workers`, the workers they never reach
    still hold their share of the semaphore budget, so the run delivers less
    concurrency than configured and nothing in the report says so.

    The smallest such case is one pin: `shared_prefix` with multi-turn on
    and `num_groups: 1` sends every request to worker 0. With two workers
    and `concurrency_level: 8`, an even split would leave worker 0 holding 4.

    One prompt per request, so each request is round 0 of its own session:
    a later round of a session blocks inside the worker until the earlier
    round finishes, and that wait would show up as under-delivery for a
    reason unrelated to routing.
    """
    concurrency, num_workers = 8, 2
    model_name = TEST_MODEL_NAME
    model_path = extract_tarball(TEST_MODEL_TARBALL)
    num_requests = concurrency * ROUNDS
    data = {
        "type": "shared_prefix",
        "shared_prefix": {
            "num_groups": 1,
            "num_prompts_per_group": num_requests,
            "system_prompt_len": 16,
            "question_len": 16,
            "output_len": OUTPUT_TOKENS,
            "enable_multi_turn_chat": True,
        },
    }

    async with LLMDInferenceSimRunner(model_name, *_pinned_latency_sim_args(), port=get_free_port()) as sim:
        result = await run_benchmark_minimal(
            _concurrent_stage_config(sim, model_name, model_path, data, concurrency, num_workers, num_requests),
            timeout_sec=180,
        )

    entries = assert_successful_run(result, num_requests)
    assert_delivered_concurrency(entries, concurrency)


# --- Helper self-tests: prove the assertions can actually fail. -------------
# Same guard as the golden accuracy suite: a load-shape assertion that cannot
# go red is worse than no assertion, because it reads as coverage. These run
# without the sim, so the reasoning stays checked even where the binary is
# absent.


# `delivered` requests in flight for `rounds` rounds, each taking `duration`.
# Round r has all `delivered` requests start at r*duration and end at
# (r+1)*duration, so in-flight is exactly `delivered` throughout.
def _closed_loop_entries(delivered: int, rounds: int, duration: float = 1.0) -> list[dict]:
    return [{"start_time": r * duration, "end_time": (r + 1) * duration} for r in range(rounds) for _ in range(delivered)]


# `n` requests spread evenly over `duration` seconds, one every duration/n.
def _even_entries(n: int, duration: float) -> list[dict]:
    return [{"start_time": i * duration / n, "end_time": i * duration / n + 0.1} for i in range(n)]


# Pins rate_tolerance at known n: poisson 2500 -> 8% (4/sqrt(n)), constant
# 2500 -> the 5% floor, poisson 10000 -> the floor too. Rejects an unknown
# arrival name and n=1.
def test_rate_tolerance_is_a_function_of_n_and_arrival():
    # Poisson tightens as 1/sqrt(n), constant as 1/n, so at equal n the
    # Poisson budget is the looser of the two.
    assert rate_tolerance(2_500, "poisson") == pytest.approx(0.08)  # 4/sqrt(n), above the floor
    assert rate_tolerance(2_500, "constant") == pytest.approx(0.05)  # 12/n is below the floor here
    assert rate_tolerance(2_500, "poisson") > rate_tolerance(2_500, "constant")
    assert rate_tolerance(10_000, "poisson") < rate_tolerance(2_500, "poisson")
    # The floor is a floor for both: no request count buys a tolerance under 5%.
    assert rate_tolerance(10_000, "poisson") == pytest.approx(0.05)
    with pytest.raises(ValueError, match="unknown arrival process"):
        rate_tolerance(100, "lognormal")
    with pytest.raises(ValueError, match="meaningless"):
        rate_tolerance(1, "constant")


# 4 requests at 0, 1, 2, 3s. Pins send_duration=3.0 and achieved_rate=4/3,
# the same count-over-span arithmetic reportgen uses.
def test_observed_send_rate_matches_reportgen_arithmetic():
    entries = [{"start_time": t, "end_time": t + 0.5} for t in [0.0, 1.0, 2.0, 3.0]]
    send_duration, rate = observed_send_rate(entries)
    assert send_duration == pytest.approx(3.0)
    # 4 requests spanning 3s: this is count/span, the same edge effect
    # reportgen has, which is exactly why rate_tolerance accounts for it.
    assert rate == pytest.approx(4.0 / 3.0)


# 10 requests at 0, 1, ..., 9s in 5 slices. Pins counts [2, 2, 2, 2, 2]: the
# last start (9.0) lands in the final slice, not one past it.
def test_arrival_bin_counts_slices_the_send_window():
    entries = [{"start_time": float(t), "end_time": t + 0.5} for t in range(10)]
    assert arrival_bin_counts(entries, 5) == [2, 2, 2, 2, 2]
    with pytest.raises(ValueError, match="at least two bins"):
        arrival_bin_counts(entries, 1)


# 400 requests configured at 40 qps for 10s: one at t=0 and 399 at t=10.
# observed_send_rate reads exactly 40 qps (400/10) and passes the rate
# check; assert_arrivals_spread fails it, with 4 empty slices out of 5.
def test_arrivals_spread_rejects_a_burst_the_rate_check_accepts():
    entries = [{"start_time": 0.0, "end_time": 0.5}] + [{"start_time": 10.0, "end_time": 10.5}] * 399
    _, rate = observed_send_rate(entries)
    assert rate == pytest.approx(40.0)
    assert abs(rate - 40.0) / 40.0 <= rate_tolerance(400, "constant")
    with pytest.raises(AssertionError, match="not spread"):
        assert_arrivals_spread(entries)


# 600 requests one every 25ms over 15s. Pins that an even spread passes, and
# that the per-slice tolerance at n=600, 5 slices is 4.5/sqrt(120), about 41%.
def test_arrivals_spread_accepts_even_arrivals():
    assert_arrivals_spread(_even_entries(600, 15.0))
    assert spread_tolerance(600, 5) == pytest.approx(4.5 / 120**0.5)


# Two requests, one covering 0-3s and one 1-2s. Pins segments
# [(0,1,1), (1,2,2), (2,3,1)], peak 2, time-weighted mean 4/3 over 0-3s, and
# one third of that window spent at exactly 2.
def test_inflight_reconstruction_counts_overlap():
    segments = inflight_segments(
        [
            {"start_time": 0.0, "end_time": 3.0},
            {"start_time": 1.0, "end_time": 2.0},
        ]
    )
    assert segments == [(0.0, 1.0, 1), (1.0, 2.0, 2), (2.0, 3.0, 1)]
    assert max_inflight(segments) == 2
    assert mean_inflight(segments, (0.0, 3.0)) == pytest.approx(4.0 / 3.0)
    assert fraction_at_level(segments, (0.0, 3.0), 2) == pytest.approx(1.0 / 3.0)


# One request 0-1s and its replacement 1-2s. Pins peak in-flight 1, not 2:
# at t=1 the end is applied before the start.
def test_inflight_reconstruction_does_not_double_count_a_handoff():
    segments = inflight_segments(
        [
            {"start_time": 0.0, "end_time": 1.0},
            {"start_time": 1.0, "end_time": 2.0},
        ]
    )
    assert max_inflight(segments) == 1


# A perfect closed loop at 4 in flight for 5 rounds of 1s. Pins the plateau
# window (0.0, 4.0): the 4th earliest start is still 0.0, and the window ends
# at the last start, before the drain. One round is too few for a window.
def test_plateau_window_excludes_ramp_up_and_drain():
    entries = _closed_loop_entries(delivered=4, rounds=5)
    assert plateau_window(entries, 4) == (0.0, 4.0)
    with pytest.raises(ValueError, match="need at least"):
        plateau_window(_closed_loop_entries(delivered=4, rounds=1), 4)


# A perfect closed loop at 8 in flight, checked against a configured 8. Passes.
def test_delivered_concurrency_accepts_a_faithful_closed_loop():
    assert_delivered_concurrency(_closed_loop_entries(delivered=8, rounds=6), 8)


# A closed loop holding 7 in flight, checked against a configured 8. Fails on
# the plateau average: the #633 failure mode, where a distribution bug
# throttles the run to concurrency_level - 1 and every other e2e still passes.
def test_delivered_concurrency_rejects_under_delivery():
    with pytest.raises(AssertionError, match="averaged"):
        assert_delivered_concurrency(_closed_loop_entries(delivered=7, rounds=6), 8)


# A closed loop holding 9 in flight, checked against a configured 8. Fails on
# the peak: the semaphore is an upper bound.
def test_delivered_concurrency_rejects_over_delivery():
    with pytest.raises(AssertionError, match="peaked at"):
        assert_delivered_concurrency(_closed_loop_entries(delivered=9, rounds=6), 8)
