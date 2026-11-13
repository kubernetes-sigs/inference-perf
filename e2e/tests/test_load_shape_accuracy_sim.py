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

import statistics

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


# Sim flags for the concurrency tests: --time-to-first-token 150ms and
# --inter-token-latency 15ms with zero jitter. Request duration must not vary,
# or the length of the plateau moves with it.
def _pinned_latency_sim_args() -> list[str]:
    return _sim_args(
        # The sim parses latencies as Go durations. A bare number fails to start it.
        *("--time-to-first-token", f"{SIM_TTFT_MS}ms"),
        *("--inter-token-latency", f"{SIM_ITL_MS}ms"),
        *("--time-to-first-token-std-dev", "0"),
        *("--inter-token-latency-std-dev", "0"),
        *("--seed", "42"),
    )


# One concurrent stage of `num_requests` requests at `concurrency` across
# `num_workers` workers, with the given `data:` block and per-request output.
# A list of counts gives a multi-stage run, every stage at the same concurrency.
def _concurrent_stage_config(
    sim: LLMDInferenceSimRunner,
    model_name: str,
    model_path: str,
    data: dict,
    concurrency: int,
    num_workers: int,
    num_requests: int | list[int],
) -> dict:
    requests_per_stage = [num_requests] if isinstance(num_requests, int) else num_requests
    return {
        "data": data,
        "load": {
            "type": "concurrent",
            "stages": [{"num_requests": count, "concurrency_level": concurrency} for count in requests_per_stage],
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

    1. reportgen's `achieved_rate` is the gaps over the span of send times,
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


@pytest.mark.asyncio
@pytest.mark.skipif(not LLMDInferenceSimRunner.is_available(), reason="local environment missing llm-d-inference-sim")
async def test_conversation_replay_replenishes_ready_conversations():
    """Conversation turns refill their slots without lock waits or rollover overlap."""
    # A remainder split also proves preferred-worker routing keeps each slot
    # owned by one worker when conversation count is not divisible by workers.
    concurrency, num_workers = 5, 2
    model_name = TEST_MODEL_NAME
    model_path = extract_tarball(TEST_MODEL_TARBALL)
    num_requests = concurrency * 8
    data = {
        "type": "conversation_replay",
        "conversation_replay": {
            "seed": 42,
            "num_conversations": concurrency,
            "shared_system_prompt_len": 16,
            "turns_per_conversation": {"type": "fixed", "min": 4, "max": 4, "mean": 4},
            "input_tokens_per_turn": {"type": "fixed", "min": 16, "max": 16, "mean": 16},
            # Heterogeneous request durations force conversations to become ready
            # out of the fixed queue order that triggered #906.
            "output_tokens_per_turn": {"type": "uniform", "min": 4, "max": 32, "mean": 18},
            "tool_call_latency_sec": {"type": "fixed", "min": 1, "max": 1, "mean": 1},
            "max_model_len": 4096,
        },
    }

    async with LLMDInferenceSimRunner(model_name, *_pinned_latency_sim_args(), port=get_free_port()) as sim:
        result = await run_benchmark_minimal(
            _concurrent_stage_config(sim, model_name, model_path, data, concurrency, num_workers, num_requests),
            timeout_sec=180,
        )

    entries = assert_successful_run(result, num_requests)
    assert_delivered_concurrency(entries, concurrency)

    by_slot: dict[int, list[dict]] = {}
    for entry in entries:
        session_id = entry["info"]["extra_info"]["user_session"]
        if session_id.startswith("conv_"):
            slot = int(session_id.removeprefix("conv_"))
        else:
            # Rollover ids are stage_{stage}_slot_{slot}_convo_{n}.
            slot = int(session_id.split("_")[3])
        by_slot.setdefault(slot, []).append(entry)

    assert set(by_slot) == set(range(concurrency))
    assert any(len(slot_entries) > 4 for slot_entries in by_slot.values()), "expected at least one conversation rollover"

    gaps = []
    for slot_entries in by_slot.values():
        ordered = sorted(slot_entries, key=lambda entry: entry["start_time"])
        for previous, current in zip(ordered, ordered[1:], strict=False):
            gap = current["start_time"] - previous["end_time"]
            assert gap >= 0, f"conversation slot overlapped across rollover by {-gap:.3f}s"
            gaps.append(gap)

    assert gaps
    assert max(gaps) < 0.1, f"slowest conversation handoff took {max(gaps):.3f}s"
    assert sum(gaps) / len(gaps) < 0.2, f"conversation handoffs averaged {sum(gaps) / len(gaps):.3f}s"


# --- Multi-stage conversation replay (#929). --------------------------------
# #701 fixed a hang that only shows up at a stage boundary, and changed what
# stage 1+ requests carry (system prompt and history, where main sent the bare
# turn). Both were proven by a manual 3-stage run and by nothing in CI, which is
# what this test pins.
STAGES = 3
CONVERSATIONS = 4
NUM_WORKERS = 2
TURNS_PER_CONVERSATION = 3
ROUNDS_PER_STAGE = 2
SHARED_SYSTEM_PROMPT_LEN = 512
INPUT_TOKENS_PER_TURN = 8
CONVERSATION_OUTPUT_TOKENS = 8
REQUESTS_PER_STAGE = CONVERSATIONS * TURNS_PER_CONVERSATION * ROUNDS_PER_STAGE
EXPECTED_CONVERSATION_REQUESTS = REQUESTS_PER_STAGE * STAGES
CONVERSATIONS_PER_STAGE = REQUESTS_PER_STAGE // TURNS_PER_CONVERSATION
# A stage-boundary hang returns nothing: no traceback, no report, no exit. It
# has to fail on a clock instead of riding out the harness's own timeout, so the
# run gets one of its own. The shape below takes about 20s on the sim.
CONVERSATION_TIMEOUT_SEC = 180
# Stages 1+ regenerate the shared system prompt from a stage-derived seed, so
# its text -- and with it the exact token count -- differs from stage 0's even
# though its length does not. The band is for that drift, not for slop: #701's
# bug moved the count by two orders of magnitude.
FIRST_TURN_TOKEN_TOLERANCE = 0.05
FIRST_TURN_TOKEN_SLACK = 8


# A per-request report entry, cut down to the fields the conversation
# assertions below read.
def _conversation_entry(stage_id: int, session_id: str, start_time: float, input_tokens: int) -> dict:
    return {
        "stage_id": stage_id,
        "start_time": start_time,
        "end_time": start_time + 0.1,
        "info": {
            "extra_info": {"user_session": session_id},
            "request_metrics": {"text": {"input_tokens": input_tokens}},
        },
    }


# (stage, session id) -> how many requests that conversation turned. Keyed by
# stage as well as session because a stage re-primes its slots under the id the
# previous stage rolled them over to, so one id names a conversation in each.
def _turns_per_conversation(entries: list[dict]) -> dict[tuple[int, str], int]:
    counts: dict[tuple[int, str], int] = {}
    for entry in entries:
        key = (int(entry["stage_id"]), str(entry["info"]["extra_info"]["user_session"]))
        counts[key] = counts.get(key, 0) + 1
    return counts


# stage id -> the input tokens of every conversation's first turn in it, in
# start order. Turns are serialized by the session lock, so a conversation's
# earliest-starting request is its turn 0.
def _first_turn_input_tokens(entries: list[dict]) -> dict[int, list[int]]:
    earliest: dict[tuple[int, str], dict] = {}
    for entry in entries:
        key = (int(entry["stage_id"]), str(entry["info"]["extra_info"]["user_session"]))
        held = earliest.get(key)
        if held is None or entry["start_time"] < held["start_time"]:
            earliest[key] = entry

    by_stage: dict[int, list[dict]] = {}
    for (stage_id, _), entry in earliest.items():
        by_stage.setdefault(stage_id, []).append(entry)

    return {
        stage_id: [
            int(entry["info"]["request_metrics"]["text"]["input_tokens"])
            for entry in sorted(stage, key=lambda e: e["start_time"])
        ]
        for stage_id, stage in sorted(by_stage.items())
    }


# Every conversation must turn exactly TURNS_PER_CONVERSATION times, and every
# stage must run CONVERSATIONS_PER_STAGE of them. A conversation longer than its
# blueprint means session state leaked across a rollover; a shorter one means a
# stage boundary cut it off. Returns {stage: conversations}.
def assert_conversations_match_blueprint(entries: list[dict]) -> dict[int, int]:
    turns = _turns_per_conversation(entries)
    misfit = {key: count for key, count in turns.items() if count != TURNS_PER_CONVERSATION}
    assert not misfit, (
        f"every conversation must turn exactly {TURNS_PER_CONVERSATION} times; these (stage, session) "
        f"pairs ran longer or shorter: {misfit}"
    )

    per_stage: dict[int, int] = {}
    for stage_id, _ in turns:
        per_stage[stage_id] = per_stage.get(stage_id, 0) + 1

    assert sorted(per_stage) == list(range(STAGES)), (
        f"expected a conversation in each of stages {list(range(STAGES))}, got {sorted(per_stage)}"
    )
    for stage_id, count in per_stage.items():
        assert count == CONVERSATIONS_PER_STAGE, (
            f"stage {stage_id} ran {count} conversations, expected {CONVERSATIONS_PER_STAGE} "
            f"({REQUESTS_PER_STAGE} requests / {TURNS_PER_CONVERSATION} turns each)"
        )
    return per_stage


# A conversation's first turn carries the stage's system prompt plus one turn's
# input. Stage 1+ must carry it exactly as stage 0 does: before #701 a slot
# re-primed at the stage boundary sent the bare turn, roughly
# INPUT_TOKENS_PER_TURN tokens against stage 0's few hundred.
def assert_first_turns_match_stage_zero(entries: list[dict]) -> None:
    first_turns = _first_turn_input_tokens(entries)
    assert 0 in first_turns, "stage 0 sent no requests, so there is no baseline to compare against"
    baseline = int(statistics.median(first_turns[0]))

    # Guards the comparison below from passing vacuously: were the system prompt
    # missing everywhere, every stage would agree on the bare turn.
    assert baseline > 4 * INPUT_TOKENS_PER_TURN, (
        f"stage 0's first turn sent {baseline} input tokens, barely more than the {INPUT_TOKENS_PER_TURN}-token "
        f"turn prompt, so the {SHARED_SYSTEM_PROMPT_LEN}-token system prompt never made it into the request"
    )

    allowed = max(FIRST_TURN_TOKEN_SLACK, FIRST_TURN_TOKEN_TOLERANCE * baseline)
    for stage_id in range(1, STAGES):
        values = first_turns.get(stage_id, [])
        assert values, f"stage {stage_id} started no conversation"
        for value in values:
            assert abs(value - baseline) <= allowed, (
                f"stage {stage_id} first turn sent {value} input tokens against stage 0's {baseline} "
                f"(drift allowed {allowed:.0f}): the system prompt did not survive the stage boundary"
            )


@pytest.mark.asyncio
@pytest.mark.skipif(not LLMDInferenceSimRunner.is_available(), reason="local environment missing llm-d-inference-sim")
async def test_conversation_replay_survives_three_stage_boundaries():
    """A conversation_replay run must carry its sessions across two stage boundaries.

    Three stages, every request delivered, no conversation longer than its
    blueprint, and stages 1 and 2 starting their conversations with the same
    first-turn prompt stage 0 does.
    """
    model_name = TEST_MODEL_NAME
    model_path = extract_tarball(TEST_MODEL_TARBALL)
    data = {
        "type": "conversation_replay",
        "conversation_replay": {
            "seed": 42,
            "num_conversations": CONVERSATIONS,
            # Fixed per-turn sizes keep every conversation's token shape the same
            # in every stage, which is what the first-turn comparison needs: only
            # the regenerated system prompt text differs.
            "shared_system_prompt_len": SHARED_SYSTEM_PROMPT_LEN,
            "turns_per_conversation": {
                "type": "fixed",
                "min": TURNS_PER_CONVERSATION,
                "max": TURNS_PER_CONVERSATION,
                "mean": TURNS_PER_CONVERSATION,
            },
            "input_tokens_per_turn": {
                "type": "fixed",
                "min": INPUT_TOKENS_PER_TURN,
                "max": INPUT_TOKENS_PER_TURN,
                "mean": INPUT_TOKENS_PER_TURN,
            },
            "output_tokens_per_turn": {
                "type": "fixed",
                "min": CONVERSATION_OUTPUT_TOKENS,
                "max": CONVERSATION_OUTPUT_TOKENS,
                "mean": CONVERSATION_OUTPUT_TOKENS,
            },
            "max_model_len": 4096,
        },
    }

    async with LLMDInferenceSimRunner(model_name, *_pinned_latency_sim_args(), port=get_free_port()) as sim:
        result = await run_benchmark_minimal(
            _concurrent_stage_config(
                sim,
                model_name,
                model_path,
                data,
                CONVERSATIONS,
                NUM_WORKERS,
                [REQUESTS_PER_STAGE] * STAGES,
            ),
            timeout_sec=CONVERSATION_TIMEOUT_SEC,
        )

    # The #701 failure mode is a stage that never finishes: before the fix the
    # run never returns from stage 2, with no traceback and no report. Fail on the
    # clock first, so that shows up here as a timeout with the log attached.
    assert not result.timed_out, (
        f"the run did not finish within {CONVERSATION_TIMEOUT_SEC}s: a hang at a stage boundary (#701) "
        f"looks exactly like this. Last CLI output:\n{result.stdout[-2000:]}"
    )

    entries = assert_successful_run(result, EXPECTED_CONVERSATION_REQUESTS)
    assert result.reports is not None

    # The boundary was crossed three times: every stage reported its own load and
    # logged completion. A stage that was skipped, or that had to terminate a
    # worker wedged on the barrier, shows up here first.
    for stage_id in range(STAGES):
        assert f"Stage {stage_id} - run completed" in result.stdout, (
            f"stage {stage_id} did not complete cleanly:\n{result.stdout}"
        )
        stage_report_name = f"stage_{stage_id}_lifecycle_metrics.json"
        assert stage_report_name in result.reports, f"missing {stage_report_name} in {sorted(result.reports)}"
        stage_report = result.reports[stage_report_name]
        assert stage_report["successes"]["count"] == REQUESTS_PER_STAGE
        assert stage_report["failures"]["count"] == 0

    assert_conversations_match_blueprint(entries)
    assert_first_turns_match_stage_zero(entries)


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


# 4 requests at 0, 1, 2, 3s. Pins send_duration=3.0 and achieved_rate=3/3,
# the same gaps-over-span arithmetic reportgen uses.
def test_observed_send_rate_matches_reportgen_arithmetic():
    entries = [{"start_time": t, "end_time": t + 0.5} for t in [0.0, 1.0, 2.0, 3.0]]
    send_duration, rate = observed_send_rate(entries)
    assert send_duration == pytest.approx(3.0)
    # 4 requests spanning 3s are 3 gaps, so the 1.0/s they were sent at.
    assert rate == pytest.approx(1.0)


# 10 requests at 0, 1, ..., 9s in 5 slices. Pins counts [2, 2, 2, 2, 2]: the
# last start (9.0) lands in the final slice, not one past it.
def test_arrival_bin_counts_slices_the_send_window():
    entries = [{"start_time": float(t), "end_time": t + 0.5} for t in range(10)]
    assert arrival_bin_counts(entries, 5) == [2, 2, 2, 2, 2]
    with pytest.raises(ValueError, match="at least two bins"):
        arrival_bin_counts(entries, 1)


# 401 requests configured at 40 qps for 10s: one at t=0 and 400 at t=10.
# observed_send_rate reads exactly 40 qps (400 gaps / 10s) and passes the rate
# check; assert_arrivals_spread fails it, with 4 empty slices out of 5.
def test_arrivals_spread_rejects_a_burst_the_rate_check_accepts():
    entries = [{"start_time": 0.0, "end_time": 0.5}] + [{"start_time": 10.0, "end_time": 10.5}] * 400
    _, rate = observed_send_rate(entries)
    assert rate == pytest.approx(40.0)
    assert abs(rate - 40.0) / 40.0 <= rate_tolerance(401, "constant")
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


# --- Helper self-tests for the multi-stage conversation assertions. ----------
# Same rule as above: a blueprint or stage-boundary assertion that cannot go red
# reads as coverage without being any.

# What a first turn costs with the configured system prompt in front of it, used
# as the synthetic stand-in for the sim's reported token count.
SYNTHETIC_FIRST_TURN_TOKENS = 520


# `turns` requests of one conversation in one stage, starting at `start`.
def _conversation_turns(stage_id: int, session_id: str, start: float, turns: int, first_turn_tokens: int) -> list[dict]:
    return [
        _conversation_entry(stage_id, session_id, start + turn, first_turn_tokens if turn == 0 else 2 * INPUT_TOKENS_PER_TURN)
        for turn in range(turns)
    ]


# A full-shape synthetic run: STAGES stages of CONVERSATIONS_PER_STAGE
# conversations with TURNS_PER_CONVERSATION turns each, every conversation's
# first turn carrying a full system prompt. `stage_first_turns` overrides a
# stage's first-turn count so a stage can be made to look like it dropped it.
def _synthetic_multi_stage_run(stage_first_turns: dict[int, int] | None = None) -> list[dict]:
    overrides = stage_first_turns or {}
    entries: list[dict] = []
    for stage_id in range(STAGES):
        first_turn = overrides.get(stage_id, SYNTHETIC_FIRST_TURN_TOKENS)
        for conversation in range(CONVERSATIONS_PER_STAGE):
            session_id = f"stage_{stage_id}_slot_{conversation % CONVERSATIONS}_convo_{conversation}"
            start = 10.0 * stage_id + 3.0 * conversation
            entries.extend(_conversation_turns(stage_id, session_id, start, TURNS_PER_CONVERSATION, first_turn))
    return entries


# One 3-turn conversation: turn 0 at 0s carrying 520 input tokens, turns 1 and 2
# at 1s and 2s carrying 16. Pins that the helpers read turn 0, not the last turn.
def test_conversation_helpers_read_turn_zero_only():
    entries = _conversation_turns(0, "conv_0", 0.0, TURNS_PER_CONVERSATION, SYNTHETIC_FIRST_TURN_TOKENS)
    assert _turns_per_conversation(entries) == {(0, "conv_0"): TURNS_PER_CONVERSATION}
    assert _first_turn_input_tokens(entries) == {0: [SYNTHETIC_FIRST_TURN_TOKENS]}


# A stage re-primes its slots under the previous stage's rollover id, so one id
# names a conversation in two stages. Pins that they are counted apart: 3 turns
# each, never 6 in one.
def test_turns_per_conversation_keeps_stages_apart():
    session_id = "stage_0_slot_0_convo_1"
    entries = _conversation_turns(0, session_id, 0.0, TURNS_PER_CONVERSATION, SYNTHETIC_FIRST_TURN_TOKENS)
    entries += _conversation_turns(1, session_id, 10.0, TURNS_PER_CONVERSATION, SYNTHETIC_FIRST_TURN_TOKENS)
    assert _turns_per_conversation(entries) == {
        (0, session_id): TURNS_PER_CONVERSATION,
        (1, session_id): TURNS_PER_CONVERSATION,
    }
    assert _first_turn_input_tokens(entries) == {0: [SYNTHETIC_FIRST_TURN_TOKENS], 1: [SYNTHETIC_FIRST_TURN_TOKENS]}


# The full-shape run passes both assertions, and reports one conversation count
# per stage.
def test_conversation_assertions_accept_a_faithful_multi_stage_run():
    entries = _synthetic_multi_stage_run()
    assert assert_conversations_match_blueprint(entries) == {stage_id: CONVERSATIONS_PER_STAGE for stage_id in range(STAGES)}
    assert_first_turns_match_stage_zero(entries)


# Stage 0's and stage 2's first turns stay at 520, stage 1's tokenize to 512 and
# 528: inside the 26-token band (5% of 520), so the run still passes.
def test_first_turn_check_allows_the_tokenizer_drift_between_stages():
    assert_first_turns_match_stage_zero(_synthetic_multi_stage_run({1: 512, 2: 528}))


# Stage 1's first turn carries only the bare turn, 8 tokens against stage 0's
# 520: the #701 regression, caught on the boundary.
def test_first_turn_check_rejects_a_stage_that_drops_the_system_prompt():
    with pytest.raises(AssertionError, match="did not survive the stage boundary"):
        assert_first_turns_match_stage_zero(_synthetic_multi_stage_run({1: INPUT_TOKENS_PER_TURN}))


# Stage 1 drifts 40 tokens off stage 0's 520, outside the 26-token band: the
# band is a band, not an absence check.
def test_first_turn_check_rejects_drift_beyond_the_band():
    with pytest.raises(AssertionError, match="did not survive the stage boundary"):
        assert_first_turns_match_stage_zero(_synthetic_multi_stage_run({1: 480}))


# No stage carries the system prompt: the baseline guard fires instead of the
# stages agreeing on the bare turn and the comparison passing vacuously.
def test_first_turn_check_refuses_a_run_with_no_system_prompt_anywhere():
    with pytest.raises(AssertionError, match="never made it into the request"):
        assert_first_turns_match_stage_zero(_synthetic_multi_stage_run({0: INPUT_TOKENS_PER_TURN}))


# Stage 1's first conversation gains a fourth turn, against a blueprint of 3:
# the check names the offending (stage, session) pair.
def test_blueprint_check_rejects_a_conversation_that_runs_past_its_turns():
    entries = _synthetic_multi_stage_run()
    entries.append(_conversation_entry(1, "stage_1_slot_0_convo_0", 99.0, SYNTHETIC_FIRST_TURN_TOKENS))
    with pytest.raises(AssertionError, match="ran longer or shorter"):
        assert_conversations_match_blueprint(entries)


# The last turn of stage 2's first conversation is missing (entries 48-50 of the
# synthetic run are its three turns): 2 turns against a blueprint of 3, the shape
# a boundary that cuts a conversation short leaves behind.
def test_blueprint_check_rejects_a_conversation_cut_short_at_a_boundary():
    entries = _synthetic_multi_stage_run()
    del entries[50]
    with pytest.raises(AssertionError, match="ran longer or shorter"):
        assert_conversations_match_blueprint(entries)
