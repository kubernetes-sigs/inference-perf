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
End-to-end verification of per-request deadline SLO headers (api.slo_request_latency_header)
flowing through openAIModelServerClientSession into summary_lifecycle_metrics.json,
stage_0_lifecycle_metrics.json, and per_request_lifecycle_metrics.json.
"""

import json
from pathlib import Path

import pytest

from utils.benchmark import run_benchmark_minimal
from utils.llm_d_inference_sim import LLMDInferenceSimRunner
from utils.net import get_free_port
from utils.testdata import extract_tarball

TEST_MODEL_NAME = "google/gemma-3-270m"
TEST_MODEL_TARBALL = "e2e/testdata/models/google_gemma-3-270m.tar.gz"
SIM_TTFT_SEC = 0.2


# Send 6 unary completion requests with a 10s deadline header against a 200ms echo sim.
# Verifies request_latency_slo_sec flows into per-request computed_metrics and summary goodput_metrics (100% attainment).
@pytest.mark.asyncio
@pytest.mark.skipif(not LLMDInferenceSimRunner.is_available(), reason="local environment missing llm-d-inference-sim")
async def test_deadline_slo_header_met_flows_to_summary_goodput(tmp_path: Path) -> None:
    dataset = tmp_path / "sharegpt.json"
    dataset.write_text(
        json.dumps([{"id": "1", "conversations": [{"from": "human", "value": "Testing"}, {"from": "gpt", "value": "OK"}]}])
    )
    model_path = extract_tarball(TEST_MODEL_TARBALL)

    async with LLMDInferenceSimRunner(
        TEST_MODEL_NAME, "--mode", "echo", "--time-to-first-token", f"{SIM_TTFT_SEC}s", port=get_free_port()
    ) as sim:
        result = await run_benchmark_minimal(
            {
                "data": {"type": "shareGPT", "path": str(dataset)},
                "load": {"type": "constant", "stages": [{"rate": 3, "duration": 2}]},
                "api": {
                    "type": "completion",
                    "streaming": False,
                    "slo_unit": "s",
                    "slo_request_latency_header": "x-llm-d-request-timeout-seconds",
                    "headers": {"x-llm-d-request-timeout-seconds": "10"},
                },
                "server": {
                    "type": "vllm",
                    "model_name": TEST_MODEL_NAME,
                    "base_url": f"http://{sim.host}:{sim.port}",
                    "ignore_eos": False,
                },
                "tokenizer": {"pretrained_model_name_or_path": str(model_path)},
                "report": {
                    "request_lifecycle": {
                        "summary": True,
                        "per_stage": True,
                        "per_request": True,
                        "per_request_fields": {"computed_metrics": True},
                    },
                },
            }
        )

    assert result.success, f"Benchmark failed: {result.stdout}"
    records = result.reports["per_request_lifecycle_metrics.json"]
    assert len(records) == 6
    for record in records:
        assert record["computed_metrics"]["request_latency_slo_sec"] == pytest.approx(10.0)

    summary = result.reports["summary_lifecycle_metrics.json"]
    goodput = summary["successes"]["goodput_metrics"]
    assert goodput is not None
    assert goodput["submitted_requests"] == 6
    assert goodput["total_requests"] == 6
    assert goodput["good_requests"] == 6
    assert goodput["goodput_percentage"] == pytest.approx(100.0)
    assert goodput["submitted_attainment_percentage"] == pytest.approx(100.0)
    assert goodput["request_latency_attainment_percentage"] == pytest.approx(100.0)
    assert goodput["good_output_tokens"] > 0
    assert goodput["output_token_goodput"] > 0.0
    assert goodput["request_goodput"] == pytest.approx(6 / summary["benchmark_time_seconds"])


# Send 6 unary completion requests with a 50ms deadline header against a 200ms echo sim.
# Every request misses the deadline, producing 0% attainment and 0.0 output_token_goodput in summary_lifecycle_metrics.json.
@pytest.mark.asyncio
@pytest.mark.skipif(not LLMDInferenceSimRunner.is_available(), reason="local environment missing llm-d-inference-sim")
async def test_deadline_slo_header_missed_reports_zero_attainment(tmp_path: Path) -> None:
    dataset = tmp_path / "sharegpt.json"
    dataset.write_text(
        json.dumps([{"id": "1", "conversations": [{"from": "human", "value": "Testing"}, {"from": "gpt", "value": "OK"}]}])
    )
    model_path = extract_tarball(TEST_MODEL_TARBALL)

    async with LLMDInferenceSimRunner(
        TEST_MODEL_NAME, "--mode", "echo", "--time-to-first-token", f"{SIM_TTFT_SEC}s", port=get_free_port()
    ) as sim:
        result = await run_benchmark_minimal(
            {
                "data": {"type": "shareGPT", "path": str(dataset)},
                "load": {"type": "constant", "stages": [{"rate": 3, "duration": 2}]},
                "api": {
                    "type": "completion",
                    "streaming": False,
                    "slo_unit": "s",
                    "slo_request_latency_header": "x-llm-d-request-timeout-seconds",
                    "headers": {"x-llm-d-request-timeout-seconds": "0.05"},
                },
                "server": {
                    "type": "vllm",
                    "model_name": TEST_MODEL_NAME,
                    "base_url": f"http://{sim.host}:{sim.port}",
                    "ignore_eos": False,
                },
                "tokenizer": {"pretrained_model_name_or_path": str(model_path)},
                "report": {
                    "request_lifecycle": {
                        "summary": True,
                        "per_stage": True,
                        "per_request": False,
                    },
                },
            }
        )

    assert result.success, f"Benchmark failed: {result.stdout}"
    goodput = result.reports["summary_lifecycle_metrics.json"]["successes"]["goodput_metrics"]
    assert goodput is not None
    assert goodput["submitted_requests"] == 6
    assert goodput["total_requests"] == 6
    assert goodput["good_requests"] == 0
    assert goodput["goodput_percentage"] == pytest.approx(0.0)
    assert goodput["submitted_attainment_percentage"] == pytest.approx(0.0)
    assert goodput["request_latency_attainment_percentage"] == pytest.approx(0.0)
    assert goodput["good_output_tokens"] == 0
    assert goodput["output_token_goodput"] == pytest.approx(0.0)
