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
reconstruction of delivered rate and in-flight concurrency computed in
`utils.load_shape` from the raw per-request `start_time`/`end_time` pairs.
The reconstruction does not read reportgen's summary numbers, so the
reported `achieved_rate` can be checked against it rather than merely
restated.

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
    assert_delivered_concurrency,
    fraction_at_level,
    inflight_segments,
    max_inflight,
    mean_inflight,
    observed_send_rate,
    plateau_window,
    rate_tolerance,
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


def _sim_args(*extra: str) -> list[str]:
    return [
        # Default max-num-seqs is 5. Server-side queueing does not change what
        # the client offers, but it does stretch request duration and with it
        # the length of the run, so keep every request scheduled immediately.
        *("--max-num-seqs", "64"),
        *extra,
    ]


def _server_block(sim: LLMDInferenceSimRunner, model_name: str) -> dict:
    return {
        "type": "vllm",
        "model_name": model_name,
        "base_url": f"http://{sim.host}:{sim.port}",
        "ignore_eos": True,
    }


@pytest.mark.asyncio
@pytest.mark.skipif(not LLMDInferenceSimRunner.is_available(), reason="local environment missing llm-d-inference-sim")
@pytest.mark.parametrize("arrival", ["constant", "poisson"])
async def test_achieved_rate_matches_configured_rate(arrival: str):
    """A fixed-rate stage must deliver the rate it was configured with.

    Two separate claims, deliberately not collapsed into one:

    1. reportgen's `achieved_rate` is the count over the span of send times,
       recomputed here from the raw per-request timestamps. Checking the
       reported value against the independent recomputation is what makes
       this more than a restatement of the tool's own summary.
    2. that rate matches the configured `rate` within a tolerance derived
       from the arrival process (see `rate_tolerance`), not from a number
       chosen after watching a run.
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

    Note what the report does and does not give us here. `load_summary`
    carries a `concurrency` field, but reportgen copies it straight off the
    stage config, so asserting on it proves only that the value was threaded
    through. The delivered value has to be reconstructed from the per-request
    start and end timestamps; that reconstruction is the actual oracle and
    `assert_delivered_concurrency` carries the reasoning for its window.

    `achieved_rate` is not asserted for this load type: main.py rewrites a
    concurrent stage to rate=num_requests, duration=1 to enqueue everything at
    once, so `requested_rate` on a concurrent stage is a dispatch detail and
    not a load-shape claim.
    """
    model_name = TEST_MODEL_NAME
    model_path = extract_tarball(TEST_MODEL_TARBALL)
    num_requests = concurrency * ROUNDS

    async with LLMDInferenceSimRunner(
        model_name,
        *_sim_args(
            *("--time-to-first-token", str(SIM_TTFT_MS)),
            *("--inter-token-latency", str(SIM_ITL_MS)),
            # Deterministic sleeps: request duration must not jitter, or the
            # length of the plateau moves with it.
            *("--time-to-first-token-std-dev", "0"),
            *("--inter-token-latency-std-dev", "0"),
            *("--seed", "42"),
        ),
        port=get_free_port(),
    ) as sim:
        result = await run_benchmark_minimal(
            {
                "data": {
                    "type": "synthetic",
                    "input_distribution": {"type": "fixed", "min": 32, "max": 32, "mean": 32},
                    "output_distribution": {
                        "type": "fixed",
                        "min": OUTPUT_TOKENS,
                        "max": OUTPUT_TOKENS,
                        "mean": OUTPUT_TOKENS,
                    },
                },
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
            },
            timeout_sec=180,
        )

    entries = assert_successful_run(result, num_requests)

    # Wiring check only, called out as such: this field is an echo of config.
    stage = result.reports["stage_0_lifecycle_metrics.json"]["load_summary"]
    assert stage["concurrency"] == concurrency, f"stage echoed concurrency {stage['concurrency']}, configured {concurrency}"

    # The load-shape claim: what was actually in flight.
    assert_delivered_concurrency(entries, concurrency)


# --- Helper self-tests: prove the assertions can actually fail. -------------
# Same guard as the golden accuracy suite: a load-shape assertion that cannot
# go red is worse than no assertion, because it reads as coverage. These run
# without the sim, so the reasoning stays checked even where the binary is
# absent.


def _closed_loop_entries(delivered: int, rounds: int, duration: float = 1.0) -> list[dict]:
    """A perfect closed loop holding `delivered` requests in flight.

    `delivered` slots run back to back for `rounds` rounds, every request
    taking exactly `duration`, so in-flight is `delivered` throughout.
    """
    return [{"start_time": r * duration, "end_time": (r + 1) * duration} for r in range(rounds) for _ in range(delivered)]


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


def test_observed_send_rate_matches_reportgen_arithmetic():
    entries = [{"start_time": t, "end_time": t + 0.5} for t in [0.0, 1.0, 2.0, 3.0]]
    send_duration, rate = observed_send_rate(entries)
    assert send_duration == pytest.approx(3.0)
    # 4 requests spanning 3s: this is count/span, the same edge effect
    # reportgen has, which is exactly why rate_tolerance accounts for it.
    assert rate == pytest.approx(4.0 / 3.0)


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


def test_inflight_reconstruction_does_not_double_count_a_handoff():
    # One request ending exactly as its replacement starts is one slot, not
    # two: ties resolve ends before starts.
    segments = inflight_segments(
        [
            {"start_time": 0.0, "end_time": 1.0},
            {"start_time": 1.0, "end_time": 2.0},
        ]
    )
    assert max_inflight(segments) == 1


def test_plateau_window_excludes_ramp_up_and_drain():
    entries = _closed_loop_entries(delivered=4, rounds=5)
    # Fourth earliest start is still 0.0 (the pipeline fills instantly here),
    # and the window closes at the last start, before the drain.
    assert plateau_window(entries, 4) == (0.0, 4.0)
    with pytest.raises(ValueError, match="need at least"):
        plateau_window(_closed_loop_entries(delivered=4, rounds=1), 4)


def test_delivered_concurrency_accepts_a_faithful_closed_loop():
    assert_delivered_concurrency(_closed_loop_entries(delivered=8, rounds=6), 8)


def test_delivered_concurrency_rejects_under_delivery():
    # The #633 failure mode: a distribution bug throttles the run to
    # concurrency_level - 1 and every existing e2e assertion still passes.
    with pytest.raises(AssertionError, match="averaged"):
        assert_delivered_concurrency(_closed_loop_entries(delivered=7, rounds=6), 8)


def test_delivered_concurrency_rejects_over_delivery():
    with pytest.raises(AssertionError, match="peaked at"):
        assert_delivered_concurrency(_closed_loop_entries(delivered=9, rounds=6), 8)
