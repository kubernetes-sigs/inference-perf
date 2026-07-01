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
"""
End-to-end test covering a mid-run simulator outage.

The scenario: 25 requests are sent to the sim and succeed, the sim is killed,
then another 25 requests are sent and fail. A single summary report is expected
to show 25 successes and 25 failures.

Requires `llm-d-inference-sim` in PATH (see test_llm_d_inference_sim.py). If it
is missing, the test is skipped automatically.
"""

import asyncio
import logging
import re
import aiohttp
import pytest

from utils.llm_d_inference_sim import LLMDInferenceSimRunner
from utils.benchmark import run_benchmark_minimal
from utils.testdata import extract_tarball

logger = logging.getLogger(__name__)

TEST_MODEL_NAME = "google/gemma-3-270m"
TEST_MODEL_TARBALL = "e2e/testdata/models/google_gemma-3-270m.tar.gz"

# Requests per stage. Stage one succeeds against the live sim, stage two fails
# after the sim is killed.
STAGE_RATE = 5
STAGE_DURATION = 5
REQUESTS_PER_STAGE = STAGE_RATE * STAGE_DURATION

# Regex matching the sim's cumulative successful-request counter. Mirrors the
# parsing used in test_prometheus.py.
_SIM_SUCCESS_RE = re.compile(r"vllm:request_success(?:_total)?\{.*?\} (\d+)")


async def _sim_success_count(session: aiohttp.ClientSession, url: str) -> int:
    """Reads the sim's cumulative successful-request count, or 0 if unavailable."""
    async with session.get(url) as resp:
        text = await resp.text()
    match = _SIM_SUCCESS_RE.search(text)
    return int(match.group(1)) if match else 0


async def _wait_for_sim_success_count(
    host: str,
    port: int,
    target: int,
    timeout_sec: float = 120,
    poll_sec: float = 0.2,
) -> int:
    """
    Polls the sim's /metrics endpoint until it reports at least `target`
    successful requests. Synchronizing on the sim's own counter (rather than a
    fixed sleep) makes the kill deterministic regardless of benchmark startup
    time: stage one contains exactly `target` requests, so the counter plateaus
    at `target` during the inter-stage interval, giving a safe window to kill
    the sim before stage two begins.
    """
    url = f"http://{host}:{port}/metrics"
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_sec
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                count = await _sim_success_count(session, url)
            except Exception as e:
                logger.debug(f"polling sim metrics failed: {e}, retrying...")
                count = 0
            if count >= target:
                return count
            if loop.time() > deadline:
                raise TimeoutError(f"sim did not reach {target} successes within {timeout_sec}s (last={count})")
            await asyncio.sleep(poll_sec)


@pytest.mark.asyncio
@pytest.mark.skipif(not LLMDInferenceSimRunner.is_available(), reason="local environment missing llm-d-inference-sim")
async def test_summary_reports_failures_when_sim_killed_midrun():
    """
    Sends 25 requests that succeed, kills the sim, then sends 25 more that fail,
    and asserts the summary report shows 25 successes and 25 failures.
    """
    model_name = TEST_MODEL_NAME
    model_path = extract_tarball(TEST_MODEL_TARBALL)
    port = 18002

    # Two equal stages separated by a generous interval. The sim is killed
    # during the interval, after stage one's requests have all succeeded and
    # before stage two's requests are dispatched.
    load = {
        "type": "constant",
        "interval": 15,
        "stages": [
            {"rate": STAGE_RATE, "duration": STAGE_DURATION},
            {"rate": STAGE_RATE, "duration": STAGE_DURATION},
        ],
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

    sim = LLMDInferenceSimRunner(model_name, port=port)
    await sim.__aenter__()

    bench_task = asyncio.create_task(run_benchmark_minimal(config))
    try:
        # Wait for stage one's 25 requests to succeed, then kill the sim so
        # stage two fails against a dead server.
        await _wait_for_sim_success_count(sim.host, sim.port, REQUESTS_PER_STAGE)
        await sim.__aexit__(None, None, None)
        result = await bench_task
    finally:
        # Safety net: ensure the sim is dead and the benchmark is drained even
        # if the wait/kill above raised.
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

    summary_report = result.reports["summary_lifecycle_metrics.json"]
    assert summary_report, "Missing summary report"

    successes = summary_report["successes"]["count"]
    failures = summary_report["failures"]["count"]
    assert successes == REQUESTS_PER_STAGE, f"Expected {REQUESTS_PER_STAGE} successes, got {successes}"
    assert failures == REQUESTS_PER_STAGE, f"Expected {REQUESTS_PER_STAGE} failures, got {failures}"

    requests_report = result.reports["per_request_lifecycle_metrics.json"]
    assert requests_report and len(requests_report) == 2 * REQUESTS_PER_STAGE, "Unexpected number of requests in report"
