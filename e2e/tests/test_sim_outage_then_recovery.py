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
"""End-to-end test covering a simulator outage that the run recovers from.

The scenario, over three equal stages: stage 0 runs against a live sim and
succeeds, the sim is killed during the interval that follows, stage 1 runs into
a dead port and fails outright, a fresh sim is started on the same port during
the next interval, and stage 2 succeeds against it.

``test_sim_killed_midrun.py`` (#583) already covers the terminal case where the
sim never comes back. Recovery is the harder half and the one #620 calls out:
after an outage the run keeps going with reused client state, so stale state
would show up as failures leaking into the recovered stage, or as a summary
whose latency and token aggregates were polluted by the requests that failed.

What is asserted:

- failures are confined to the outage stage: stage 0 and stage 2 report zero
  failures, stage 1 reports nothing but failures;
- the post-recovery stage really succeeds, at full count, against the restarted
  sim;
- the run-wide latency and token aggregates are computed from the successful
  requests only. Recomputing them from the per-request report is the oracle,
  and the same numbers recomputed over every request (successes and failures)
  must differ, so the check cannot pass vacuously.

"Fake the conditions, never the oracle": only the outage is staged, by stopping
and restarting the real simulator. Every asserted number comes from the
generated reports, checked against the per-request records the same run
produced.

Requires `llm-d-inference-sim` in PATH (see test_llm_d_inference_sim.py). If it
is missing, the test is skipped automatically.
"""

import asyncio
import logging
import re
import statistics
from typing import Any, Dict, List, Optional

import aiohttp
import pytest

from utils.llm_d_inference_sim import LLMDInferenceSimRunner
from utils.benchmark import run_benchmark_minimal
from utils.net import get_free_port
from utils.testdata import extract_tarball

logger = logging.getLogger(__name__)

TEST_MODEL_NAME = "google/gemma-3-270m"
TEST_MODEL_TARBALL = "e2e/testdata/models/google_gemma-3-270m.tar.gz"

# Three equal stages: live, dead, live again.
STAGE_RATE = 5
STAGE_DURATION = 3
REQUESTS_PER_STAGE = STAGE_RATE * STAGE_DURATION
NUM_STAGES = 3

# A long inter-stage interval is what makes the outage line up with stage 1
# exactly. Both the kill and the restart have to land inside an interval, and
# the only observable clock the test shares with the benchmark is the sim's own
# metrics counter, which goes away with the sim.
STAGE_INTERVAL = 30

# What a stage costs on top of its configured duration: run_stage offsets
# dispatch by one second, polls for completion at one second granularity, then
# drains its queue. Rounded well up so a loaded machine still fits.
STAGE_OVERHEAD_SEC = 6

# Extra slack before the restart, on top of "stage 1 must be over by now".
# The restart has to land inside the interval after stage 1: too early and the
# revived sim catches the tail of stage 1, too late and stage 2 starts against
# a dead port. Stage 1 overhead is the only uncertain term, so the margin sits
# on the early side of the interval's midpoint rather than at its edge.
RESTART_MARGIN_SEC = 13

# Measured from the moment the kill completes. Stage 1 begins one interval
# after stage 0 ends and takes duration + overhead, so waiting that out plus
# the margin puts the restart inside the interval that follows stage 1, with
# roughly STAGE_INTERVAL - RESTART_MARGIN_SEC still to spare before stage 2
# begins.
RESTART_AFTER_KILL_SEC = STAGE_INTERVAL + STAGE_DURATION + STAGE_OVERHEAD_SEC + RESTART_MARGIN_SEC

# Floating point slack when comparing a report aggregate to the same aggregate
# recomputed from the per-request records.
RELATIVE_TOLERANCE = 1e-6

# Regex matching one series of the sim's cumulative successful-request
# counter. Same metric as test_prometheus.py and test_sim_killed_midrun.py, but
# summed over every label set below: the sim labels the counter by
# finish_reason, so reading only the first series would plateau short of the
# target whenever a stage mixes finish reasons.
_SIM_SUCCESS_RE = re.compile(r"^vllm:request_success(?:_total)?\{.*?\} (\d+)", re.MULTILINE)


# One GET of the sim's /metrics. Returns the sum of every vllm:request_success(_total)
# series (e.g. finish_reason="stop" 9 + finish_reason="length" 6 -> 15), or None if the
# counter is not on the page yet.
async def _sim_success_count(session: aiohttp.ClientSession, url: str) -> Optional[int]:
    """Reads the sim's cumulative successful-request count, summed over all
    finish_reason series. Returns None when the counter is absent: the sim
    only creates the series on the first successful request, so absence is
    normal until stage 0 starts completing and is only suspicious if it lasts."""
    async with session.get(url) as resp:
        text = await resp.text()
    counts = _SIM_SUCCESS_RE.findall(text)
    if not counts:
        return None
    return sum(int(count) for count in counts)


# Polls that sum every 0.2s until it reaches target (15 here, one full stage) and returns
# it. Gives up after 120s with one of two messages: counter seen but short of target, or
# counter never seen at all (no request completed, or the sim renamed the metric).
async def _wait_for_sim_success_count(
    host: str,
    port: int,
    target: int,
    timeout_sec: float = 120,
    poll_sec: float = 0.2,
) -> int:
    """Polls the sim's /metrics endpoint until it reports at least `target`
    successful requests. Synchronizing on the sim's own counter rather than a
    fixed sleep is what makes the kill deterministic: stage 0 contains exactly
    `target` requests, so the counter plateaus there for the whole interval
    that follows.
    """
    url = f"http://{host}:{port}/metrics"
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_sec
    last_seen: Optional[int] = None
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                count = await _sim_success_count(session, url)
            except Exception as e:
                logger.debug(f"polling sim metrics failed: {e}, retrying...")
                count = None
            if count is not None:
                last_seen = count
                if count >= target:
                    return count
            if loop.time() > deadline:
                # Two different failures share this timeout: the sim served
                # the counter but it never reached target, or the counter never
                # appeared at all, which points at a renamed metric (or a sim
                # that never answered) rather than at the load.
                if last_seen is not None:
                    raise TimeoutError(f"sim did not reach {target} successes within {timeout_sec}s (last={last_seen})")
                raise TimeoutError(
                    f"vllm:request_success(_total) never appeared in {url} within {timeout_sec}s: "
                    "either no request completed or the sim renamed the counter"
                )
            await asyncio.sleep(poll_sec)


# end_time - start_time of one per-request record, in seconds.
def _entry_latency(entry: Dict[str, Any]) -> float:
    return float(entry["end_time"]) - float(entry["start_time"])


# Output tokens for one record: server_usage.completion_tokens if present, otherwise the
# client-side output_tokens, otherwise 0. A failed request has no response_metrics, so 0.
def _entry_output_tokens(entry: Dict[str, Any]) -> float:
    """Output tokens the report would attribute to this request: the
    server-reported completion_tokens when present, the client count otherwise.
    Mirrors summarize_output_token_usage."""
    metrics = (entry.get("info") or {}).get("response_metrics") or {}
    usage = metrics.get("server_usage") or {}
    if usage.get("completion_tokens") is not None:
        return float(usage["completion_tokens"])
    return float(metrics.get("output_tokens") or 0)


# Prompt tokens for one record from request_metrics.text.input_tokens, otherwise 0.
def _entry_input_tokens(entry: Dict[str, Any]) -> float:
    request_metrics = ((entry.get("info") or {}).get("request_metrics") or {}).get("text") or {}
    return float(request_metrics.get("input_tokens") or 0)


# actual must equal expected to within 1e-6 relative; `what` names the aggregate in the message.
def _assert_close(actual: float, expected: float, what: str) -> None:
    assert actual == pytest.approx(expected, rel=RELATIVE_TOLERANCE), f"{what}: report says {actual}, recomputed {expected}"


# The opposite: polluted must NOT equal clean to within 1e-6. Fails when the failed requests
# would not have moved the number, because then the successes-only check proves nothing.
def _assert_differs(polluted: float, clean: float, what: str) -> None:
    """The teeth of the successes-only check: if including the failed requests
    made no difference, the check proves nothing about which set was used."""
    assert polluted != pytest.approx(clean, rel=RELATIVE_TOLERANCE), (
        f"{what}: including failures changes nothing, this assertion cannot detect pollution"
    )


# Splits the 45 per-request records into 30 successes and 15 failures, checks no failure
# carries response_metrics, then checks the summary's mean/max latency and total/mean output
# and prompt tokens equal the same numbers recomputed from the 30 successes, and that
# recomputing over all 45 gives different numbers.
def _assert_aggregates_use_successes_only(summary: Dict[str, Any], per_request: List[Dict[str, Any]]) -> None:
    successes = [entry for entry in per_request if not entry.get("error")]
    failures = [entry for entry in per_request if entry.get("error")]
    assert len(successes) == (NUM_STAGES - 1) * REQUESTS_PER_STAGE
    assert len(failures) == REQUESTS_PER_STAGE

    # A failed request must not carry response metrics at all: that is the
    # structural reason its latency and tokens cannot reach the aggregates.
    for entry in failures:
        assert not (entry.get("info") or {}).get("response_metrics"), (
            f"a failed request carries response_metrics: {entry.get('error')}"
        )

    latencies = [_entry_latency(entry) for entry in successes]
    _assert_close(
        summary["successes"]["latency"]["request_latency"]["mean"],
        statistics.fmean(latencies),
        "mean request latency",
    )
    _assert_close(
        summary["successes"]["latency"]["request_latency"]["max"],
        max(latencies),
        "max request latency",
    )
    _assert_differs(
        statistics.fmean([_entry_latency(entry) for entry in per_request]),
        statistics.fmean(latencies),
        "mean request latency",
    )

    output_tokens = [_entry_output_tokens(entry) for entry in successes]
    _assert_close(summary["successes"]["output_tokens"]["total"], sum(output_tokens), "total output tokens")
    _assert_close(summary["successes"]["output_tokens"]["mean"], statistics.fmean(output_tokens), "mean output tokens")
    _assert_differs(
        statistics.fmean([_entry_output_tokens(entry) for entry in per_request]),
        statistics.fmean(output_tokens),
        "mean output tokens",
    )

    input_tokens = [_entry_input_tokens(entry) for entry in successes]
    _assert_close(summary["successes"]["prompt_tokens"]["total"], sum(input_tokens), "total prompt tokens")
    _assert_close(summary["successes"]["prompt_tokens"]["mean"], statistics.fmean(input_tokens), "mean prompt tokens")
    _assert_differs(
        statistics.fmean([_entry_input_tokens(entry) for entry in per_request]),
        statistics.fmean(input_tokens),
        "mean prompt tokens",
    )


# Three 15-request stages (5/s x 3s) 30s apart. The sim is killed once its /metrics shows
# stage 0's 15 successes and restarted 52s later, before stage 2. Expects stage reports of
# 0/15/0 failures and 15/0/15 successes, a 30-success/15-failure summary, 45 per-request
# records, and summary aggregates that match the 30 successes only.
@pytest.mark.asyncio
@pytest.mark.skipif(not LLMDInferenceSimRunner.is_available(), reason="local environment missing llm-d-inference-sim")
async def test_run_recovers_after_a_midrun_sim_outage():
    """Kills the sim after stage 0, lets stage 1 fail against the dead port,
    restarts the sim before stage 2, and asserts the failures stayed inside
    stage 1 while the aggregates stayed clean."""
    model_name = TEST_MODEL_NAME
    model_path = extract_tarball(TEST_MODEL_TARBALL)
    port = get_free_port()

    load = {
        "type": "constant",
        "interval": STAGE_INTERVAL,
        "stages": [{"rate": STAGE_RATE, "duration": STAGE_DURATION} for _ in range(NUM_STAGES)],
        "num_workers": 2,
    }

    config = {
        "data": {"type": "mock"},
        "load": load,
        "api": {
            "type": "completion",
            "streaming": True,
        },
        "server": {
            "type": "vllm",
            "model_name": model_name,
            "base_url": f"http://127.0.0.1:{port}",
            "ignore_eos": True,
        },
        "tokenizer": {
            "pretrained_model_name_or_path": str(model_path),
        },
        "report": {
            "request_lifecycle": {
                "summary": True,
                "per_stage": True,
                "per_request": True,
            },
        },
    }

    first_sim = LLMDInferenceSimRunner(model_name, port=port)
    await first_sim.__aenter__()
    second_sim = LLMDInferenceSimRunner(model_name, port=port)

    bench_task = asyncio.create_task(run_benchmark_minimal(config))
    try:
        # Stage 0's requests have all succeeded, so the run is in the interval
        # before stage 1: a safe window to take the sim away.
        await _wait_for_sim_success_count(first_sim.host, first_sim.port, REQUESTS_PER_STAGE)
        await first_sim.__aexit__(None, None, None)

        # Nothing is listening now, so there is no counter to synchronize on.
        # Wait out stage 1 and restart inside the interval that follows it.
        await asyncio.sleep(RESTART_AFTER_KILL_SEC)
        await second_sim.__aenter__()

        result = await bench_task
    finally:
        # Safety net: both sims dead and the benchmark drained even if the
        # sequence above raised part way through.
        for sim in (first_sim, second_sim):
            if sim._proc is not None and sim._proc.returncode is None:
                await sim.__aexit__(None, None, None)
        if not bench_task.done():
            bench_task.cancel()
            try:
                await bench_task
            except (asyncio.CancelledError, Exception):
                pass

    assert result.success, f"Benchmark did not complete cleanly:\n{result.stdout}"
    assert result.reports, "No reports generated from benchmark"

    # Failures confined to the outage window.
    expected_failures = {0: 0, 1: REQUESTS_PER_STAGE, 2: 0}
    for stage_id, failures in expected_failures.items():
        stage_report = result.reports.get(f"stage_{stage_id}_lifecycle_metrics.json")
        assert stage_report, f"Missing report for stage {stage_id}"
        assert stage_report["failures"]["count"] == failures, (
            f"stage {stage_id} reported {stage_report['failures']['count']} failures, expected {failures}. "
            "If stage 1 is not the only failing stage, the outage window drifted off the stage boundaries."
        )
        assert stage_report["successes"]["count"] == REQUESTS_PER_STAGE - failures, (
            f"stage {stage_id} reported {stage_report['successes']['count']} successes, "
            f"expected {REQUESTS_PER_STAGE - failures}"
        )

    summary_report = result.reports["summary_lifecycle_metrics.json"]
    assert summary_report["successes"]["count"] == (NUM_STAGES - 1) * REQUESTS_PER_STAGE
    assert summary_report["failures"]["count"] == REQUESTS_PER_STAGE

    per_request = result.reports["per_request_lifecycle_metrics.json"]
    assert len(per_request) == NUM_STAGES * REQUESTS_PER_STAGE, "Unexpected number of requests in report"

    _assert_aggregates_use_successes_only(summary_report, per_request)
