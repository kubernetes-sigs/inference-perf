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
"""Does a run survive a stage that sends zero requests?

A rate of zero is legal, and so is a rate too low to reach one request: the
stage sends nothing and lasts its duration. The unit tests check the timer,
the stage loop and the report one piece at a time. Nothing there shows that a
whole run, with worker processes and a real report set, gets through a stage
that never puts a request on the queue.

The run here puts such a stage between two normal ones against the sim and
reads the three stage reports, the partials and validation.json.
"""

from typing import Any, Dict

import pytest

from utils.benchmark import run_benchmark_minimal
from utils.llm_d_inference_sim import LLMDInferenceSimRunner
from utils.net import get_free_port
from utils.testdata import extract_tarball

TEST_MODEL_NAME = "google/gemma-3-270m"
TEST_MODEL_TARBALL = "e2e/testdata/models/google_gemma-3-270m.tar.gz"

RATE = 5
DURATION = 4
IDLE_DURATION = 3
EXPECTED_REQUESTS = RATE * DURATION


# Three stages on one worker with mock prompts: 5 req/s for 4s, the given rate
# that sends nothing for 3s (or the given stage fields, for a request_interval),
# then 5 req/s for 4s again. No load.type, so the default applies.
def _config(sim: LLMDInferenceSimRunner, model_path: str, zero_rate: object) -> Dict[str, Any]:
    return {
        "api": {"type": "completion", "streaming": True},
        "data": {"type": "mock"},
        "load": {
            "stages": [
                {"rate": RATE, "duration": DURATION},
                {**(zero_rate if isinstance(zero_rate, dict) else {"rate": zero_rate}), "duration": IDLE_DURATION},
                {"rate": RATE, "duration": DURATION},
            ],
            "interval": 0,
            "num_workers": 1,
        },
        "server": {"type": "vllm", "model_name": TEST_MODEL_NAME, "base_url": f"http://{sim.host}:{sim.port}"},
        "tokenizer": {"pretrained_model_name_or_path": model_path},
        "report": {"request_lifecycle": {"summary": True, "per_stage": True, "per_request": True}},
    }


# Stages of rate 5, a stage that sends nothing, 5 over 4s, 3s, 4s. The middle stage is the rate 0,
# the rate "Max(0, -t)", the rate 0.1, which over 3s is 0.3 of a request, and a request_interval
# of 100s, longer than the stage. The run exits 0, the
# stage reports count 20, 0 and 20, the middle stage reports its requested rate and an achieved
# rate of 0 and has a BR0.2 partial, the log says why the stage sent nothing, and validation.json
# has no errors.
@pytest.mark.asyncio
@pytest.mark.skipif(not LLMDInferenceSimRunner.is_available(), reason="local environment missing llm-d-inference-sim")
@pytest.mark.parametrize(
    ("zero_rate", "requested_rate", "reason"),
    [
        (0, 0, "is zero for the whole stage"),
        ("Max(0, -t)", 0, "is zero for the whole stage"),
        (0.1, 0.1, "adds up to 0.3 requests over 3s, which is less than one request"),
        ({"request_interval": 100}, None, "no gap of its request_interval 100.0 fits in the stage"),
    ],
)
async def test_zero_request_stage_between_two_normal_stages(
    zero_rate: object, requested_rate: float | None, reason: str
) -> None:
    model_path = str(extract_tarball(TEST_MODEL_TARBALL))
    async with LLMDInferenceSimRunner(TEST_MODEL_NAME, "--max-num-seqs", "64", port=get_free_port()) as sim:
        result = await run_benchmark_minimal(_config(sim, model_path, zero_rate), timeout_sec=180)

    assert result.success, f"run failed (rc={result.return_code}):\n{result.stdout[-4000:]}"
    assert result.reports is not None

    counts = [result.reports[f"stage_{stage}_lifecycle_metrics.json"]["load_summary"]["count"] for stage in range(3)]
    assert counts == [EXPECTED_REQUESTS, 0, EXPECTED_REQUESTS]

    idle = result.reports["stage_1_lifecycle_metrics.json"]
    if requested_rate is None:
        assert idle["load_summary"]["request_interval"] == "100.0"
    else:
        assert idle["load_summary"]["requested_rate"] == pytest.approx(requested_rate)
    assert idle["load_summary"]["achieved_rate"] == 0
    assert idle["successes"]["count"] == 0
    assert idle["failures"]["count"] == 0
    assert result.reports["summary_lifecycle_metrics.json"]["load_summary"]["count"] == 2 * EXPECTED_REQUESTS
    assert len(result.reports["per_request_lifecycle_metrics.json"]) == 2 * EXPECTED_REQUESTS
    assert "inference-perf.partial.stage_1.yaml" in result.reports

    assert "Stage 1 sends zero requests" in result.stdout
    assert reason in result.stdout

    validation = result.reports["validation.json"]
    errors = validation["global"]["errors"] + [e for group in validation["reports"].values() for e in group["errors"]]
    assert errors == []
