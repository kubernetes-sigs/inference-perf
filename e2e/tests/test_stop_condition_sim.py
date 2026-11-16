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
A stage's `stop_condition` ends it, from the config file through to the report.

Three cases against llm-d-inference-sim: a time-only condition that must send
what the same `duration` sends, a request count that ends a stage long before
its time bound, and a server that fails every request.

Requires `llm-d-inference-sim` in PATH (see test_llm_d_inference_sim.py). If it
is missing, the tests are skipped automatically.
"""

import re
from typing import Any, Dict, List

import pytest

from utils.benchmark import BenchmarkResult, run_benchmark_minimal
from utils.llm_d_inference_sim import LLMDInferenceSimRunner
from utils.testdata import extract_tarball

TEST_MODEL_NAME = "google/gemma-3-270m"
TEST_MODEL_TARBALL = "e2e/testdata/models/google_gemma-3-270m.tar.gz"

RATE = 10
# far past anything a passing run reaches, so a stage that ran to it is unmistakable
TIME_BOUND_SEC = 120
REQUESTS_AT_TIME_BOUND = RATE * TIME_BOUND_SEC

HELD_AT = re.compile(r"Stage (\d+): stop condition .* held at t=([0-9.]+)s")


# A constant-load config of the given stages at 2 workers, pointed at the sim.
# E.g. [{"rate": 10, "duration": 3}] -> one stage of 30 mock completion requests.
def _config(sim: LLMDInferenceSimRunner, stages: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "data": {"type": "mock"},
        "load": {"type": "constant", "stages": stages, "interval": 1, "num_workers": 2},
        "api": {"type": "completion", "streaming": True},
        "server": {
            "type": "vllm",
            "model_name": TEST_MODEL_NAME,
            "base_url": f"http://{sim.host}:{sim.port}",
            "ignore_eos": True,
        },
        "tokenizer": {"pretrained_model_name_or_path": str(extract_tarball(TEST_MODEL_TARBALL))},
        "report": {"request_lifecycle": {"summary": True, "per_stage": True}},
    }


# The stage second each measured condition held at, by stage, read from the run's log.
# E.g. a log line "Stage 2: stop condition '...' held at t=2.31s with ..." -> {2: 2.31}.
def _held_at(result: BenchmarkResult) -> Dict[int, float]:
    return {int(stage): float(second) for stage, second in HELD_AT.findall(result.stdout)}


# (successes, failures) of one stage's report, e.g. 30 answered and none failed -> (30, 0).
def _counts(result: BenchmarkResult, stage: int) -> tuple[int, int]:
    assert result.reports, "No reports generated from benchmark"
    report = result.reports[f"stage_{stage}_lifecycle_metrics.json"]
    return report["successes"]["count"], report["failures"]["count"]


# The report set's own validator found nothing wrong.
def _assert_reports_valid(result: BenchmarkResult) -> None:
    assert result.reports
    validation = result.reports["validation.json"]
    errors = validation["global"]["errors"] + [e for group in validation["reports"].values() for e in group["errors"]]
    assert errors == [], errors


@pytest.mark.asyncio
@pytest.mark.skipif(not LLMDInferenceSimRunner.is_available(), reason="local environment missing llm-d-inference-sim")
# Three stages at 10 requests/s against a healthy sim.
# Stage 0 `duration: 3` and stage 1 `stop_condition: "t >= 3"` both send exactly 30.
# Stage 2 `(t >= 120) | (completed >= 20)` ends on the 20th response, about 2s in, not at 120s.
async def test_stop_condition_ends_a_stage_on_time_and_on_a_request_count() -> None:
    async with LLMDInferenceSimRunner(TEST_MODEL_NAME, "--time-to-first-token", "20ms") as sim:
        result = await run_benchmark_minimal(
            _config(
                sim,
                [
                    {"rate": RATE, "duration": 3},
                    {"rate": RATE, "stop_condition": "t >= 3"},
                    {"rate": RATE, "stop_condition": f"(t >= {TIME_BOUND_SEC}) | (completed >= 20)"},
                ],
            ),
            timeout_sec=90,
        )

    assert result.success, f"Benchmark did not complete cleanly:\n{result.stdout}"

    # a time-only condition is the same stage as the duration it spells out
    assert _counts(result, 0) == (30, 0)
    assert _counts(result, 1) == (30, 0)

    # only stage 2 reads a measured term, so only it logs a second it held at
    held_at = _held_at(result)
    assert set(held_at) == {2}, held_at
    assert held_at[2] < 10, held_at

    # requests already sent when the 20th response landed still finish, the rest are never sent
    successes, failures = _counts(result, 2)
    assert failures == 0
    assert 20 <= successes < REQUESTS_AT_TIME_BOUND // 10, successes

    _assert_reports_valid(result)


@pytest.mark.asyncio
@pytest.mark.skipif(not LLMDInferenceSimRunner.is_available(), reason="local environment missing llm-d-inference-sim")
# The sim rejects every request as an invalid API key. Stage 0 `(t >= 120) | (errors >= 10)` at 10 requests/s
# ends on the tenth failure, about 1s in, with no success. Stage 1 `duration: 2` still runs its 20 requests.
async def test_stop_condition_ends_a_stage_when_every_request_fails() -> None:
    async with LLMDInferenceSimRunner(
        TEST_MODEL_NAME, "--failure-injection-rate", "100", "--failure-types", "invalid_api_key"
    ) as sim:
        result = await run_benchmark_minimal(
            _config(
                sim,
                [
                    {"rate": RATE, "stop_condition": f"(t >= {TIME_BOUND_SEC}) | (errors >= 10)"},
                    {"rate": RATE, "duration": 2},
                ],
            ),
            timeout_sec=90,
        )

    assert result.success, f"Benchmark did not complete cleanly:\n{result.stdout}"

    held_at = _held_at(result)
    assert set(held_at) == {0}, held_at
    assert held_at[0] < 10, held_at

    successes, failures = _counts(result, 0)
    assert successes == 0
    assert 10 <= failures < REQUESTS_AT_TIME_BOUND // 10, failures

    # the condition ended its own stage, not the run
    assert _counts(result, 1) == (0, 20)

    _assert_reports_valid(result)
