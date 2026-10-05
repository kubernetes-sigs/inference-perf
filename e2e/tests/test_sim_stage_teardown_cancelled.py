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
End-to-end test covering requests cancelled in flight at stage teardown.

The scenario: the sim holds every response for 3 seconds. A circuit breaker
opens after the first 5 responses, which ends the stage while later requests
are still waiting on the server. The teardown grace is 0.5 seconds, so those
requests are cancelled. The stage report is expected to show them as failures
labelled "Cancelled at Stage Teardown", and none labelled "Timeout".

Requires `llm-d-inference-sim` in PATH (see test_llm_d_inference_sim.py). If it
is missing, the test is skipped automatically.
"""

import logging

import pytest

from utils.benchmark import run_benchmark_minimal
from utils.llm_d_inference_sim import LLMDInferenceSimRunner
from utils.net import get_free_port
from utils.testdata import extract_tarball

logger = logging.getLogger(__name__)

TEST_MODEL_NAME = "google/gemma-3-270m"
TEST_MODEL_TARBALL = "e2e/testdata/models/google_gemma-3-270m.tar.gz"

# 5 requests a second for 6 seconds, each held 3 seconds by the sim. The first
# responses land about 3 seconds in, so when the breaker opens roughly 15
# requests are in flight and most have more than the grace left to run.
STAGE_RATE = 5
STAGE_DURATION = 6
SIM_RESPONSE_DELAY_MS = 3000
BREAKER_THRESHOLD = 5
TEARDOWN_GRACE_SEC = 0.5

CANCELLED_LABEL = "Cancelled at Stage Teardown"


# Sim holds each response 3s; a breaker ends the stage after 5 responses with a
# 0.5s grace. Expects the stage report to show at least 5 successes, at least one
# failure labelled "Cancelled at Stage Teardown", and no other failure label.
@pytest.mark.asyncio
@pytest.mark.skipif(not LLMDInferenceSimRunner.is_available(), reason="local environment missing llm-d-inference-sim")
async def test_requests_cancelled_at_stage_teardown_are_reported_as_failures():
    model_name = TEST_MODEL_NAME
    model_path = extract_tarball(TEST_MODEL_TARBALL)
    port = get_free_port()

    config = {
        "circuit_breakers": [
            {
                "name": "end_stage_early",
                "metrics": {"matches": ["stage_id == `0`"]},
                "triggers": [{"type": "consecutive", "threshold": BREAKER_THRESHOLD}],
            }
        ],
        "data": {"type": "mock"},
        "load": {
            "type": "constant",
            "stages": [{"rate": STAGE_RATE, "duration": STAGE_DURATION}],
            "num_workers": 2,
            "circuit_breakers": ["end_stage_early"],
            "stage_teardown_grace_seconds": TEARDOWN_GRACE_SEC,
        },
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
            },
        },
    }

    async with LLMDInferenceSimRunner(
        model_name,
        *("--time-to-first-token", str(SIM_RESPONSE_DELAY_MS)),
        port=port,
    ):
        result = await run_benchmark_minimal(config)

    assert result.success, f"Benchmark did not complete cleanly:\n{result.stdout}"
    assert result.reports, "No reports generated from benchmark"

    stage_report = result.reports["stage_0_lifecycle_metrics.json"]
    assert stage_report, "Missing stage report"

    # The breaker needs BREAKER_THRESHOLD responses to open, so at least that
    # many requests completed before the stage was cut short.
    successes = stage_report["successes"]["count"]
    assert successes >= BREAKER_THRESHOLD, f"Expected at least {BREAKER_THRESHOLD} successes, got {successes}"

    # Requests sent to the sim and cancelled when the grace ran out must be in
    # the report. How many depends on timing, so the count is only bounded below.
    by_label = {label: bucket["count"] for label, bucket in stage_report["failures"]["by_label"].items()}
    assert by_label.get(CANCELLED_LABEL, 0) >= 1, f"No request reported as cancelled at stage teardown: {by_label}"

    # They are the stage's deadline, not the request's: nothing may be filed
    # under "Timeout", and the sim itself fails nothing.
    assert set(by_label) == {CANCELLED_LABEL}, f"Unexpected failure labels: {by_label}"
    assert stage_report["failures"]["count"] == by_label[CANCELLED_LABEL]
